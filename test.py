import atexit
import sys
import os
import argparse
import subprocess
import datetime
import shutil
import tempfile
import time
import re
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable

import toml
from rich import print
from rich.table import Table
from rich.progress import track
from rich.traceback import Traceback

### Settings ###


TEST_DIR = os.path.dirname(__file__)
IR_PATH = os.path.join(TEST_DIR, "ir.py")
VENUS_JAR = os.path.join(TEST_DIR, "venus.jar")
COVERAGE_PATH = os.path.join(TEST_DIR, "coverage.py")
PYTHON = sys.executable  # always use the current python
JAVA = shutil.which("java")
CC = shutil.which("gcc") or shutil.which("clang")
RV32_GCC = shutil.which("riscv32-unknown-elf-gcc")
RV32_QEMU = shutil.which("qemu-riscv32")

TEMP_DIR = tempfile.mkdtemp()
atexit.register(shutil.rmtree, TEMP_DIR)

@dataclass
class Config:
    verbose: bool
    use_qemu: bool
    parallel: bool
    check_ssa: bool
    timeout: int
    extra_cflags: list[str]
    
cfg = Config(
    verbose = False,
    use_qemu = False,
    parallel = True,
    check_ssa = False,
    timeout = 10,
    extra_cflags = []
)

### Color Utils ###


def red(s: str) -> str:
    return f"[bold red]{s}[/bold red]"

def green(s: str) -> str:
    return f"[bold green]{s}[/bold green]"

def blue(s: str) -> str:
    return f"[bold blue]{s}[/bold blue]"

def yellow(s: str) -> str:
    return f"[bold yellow]{s}[/bold yellow]"

def box(s: str) -> str:
    return f"[bold reverse white on black]{s}[/bold reverse white on black]"

### Test Utils ###


@dataclass
class Test:
    filename: str
    inputs: list[str] | None
    expected: list[str] | None
    should_fail: bool

    @staticmethod
    def parse_file(filename: str) -> "Test":
        content = open(filename).readlines()
        comments = []
        for line in content:
            # get comment, start with //
            if line.strip().startswith("//"):
                comments.append(line.strip()[2:])
            else:
                break
        if len(comments) == 0:  # no comment means success
            return Test(filename, None, None, False)
        elif len(comments) == 1:
            if "Error" in comments[0]:  # should fail
                return Test(filename, None, None, True)
            else:
                return Test(filename, None, None, False)
        elif len(comments) == 2:  # input and output
            assert "Input:" in comments[0], f"{filename} has non-paired input/output"
            assert "Output:" in comments[1], f"{filename} has non-paired input/output"
            input = comments[0].replace("Input:", "").split()
            if input[0] == "None":
                input = None
            expected = comments[1].replace("Output:", "").split()
            if expected[0] == "None":
                expected = None
            return Test(filename, input, expected, False)
        else:
            assert False, f"{filename} heading comment is invalid"

    def __str__(self):
        return f"Test({self.filename}, {self.inputs}, {self.expected}, {self.should_fail})"

class ResultType(Enum):
    ACCEPTED = auto()
    WRONG_ANSWER = auto()
    RUN_TIMEOUT = auto()
    COMPILE_TIMEOUT = auto()
    COMPILE_ERROR = auto()
    RUNTIME_ERROR = auto()

class TestResult:
    def __init__(self, test: Test, output: str | None | list[str] = None, run_ret_code: int | None = None, result_type: ResultType | None = None, concat_output: bool = False, run_time: float | None = None, run_step: int | None = None):
        self.test = test
        self.output = output
        self.run_time = run_time
        self.run_step = run_step
        if test.should_fail:
            self.result = ResultType.ACCEPTED if result_type == ResultType.COMPILE_ERROR else ResultType.WRONG_ANSWER
        else:
            if run_ret_code is not None and run_ret_code != 0:
                assert result_type is None, "run_ret_code and result_type cannot be both not None."
                self.result = ResultType.RUNTIME_ERROR
            elif result_type is not None and result_type != ResultType.ACCEPTED:
                self.result = result_type
            else:
                if test.expected is None:
                    self.result = ResultType.WRONG_ANSWER if output else ResultType.ACCEPTED
                else:
                    if not concat_output:  # lab3
                        self.result = ResultType.ACCEPTED if output == test.expected else ResultType.WRONG_ANSWER
                    else:  # lab4
                        expected = "".join(test.expected)
                        self.result = ResultType.ACCEPTED if output == expected else ResultType.WRONG_ANSWER


def measure_execution_time(func: Callable, *args, repeat: int = 10, use_last: int = 5, **kwargs) -> tuple[Any, float]:
    """
    Measure the execution time of a Callable object and return its result and average runtime.

    Args:
        func (Callable): The function to measure (no parameters).
        repeat (int): Total number of repetitions, default is 10.
        use_last (int): Number of last executions to use for average calculation, default is 5.

    Returns:
        Tuple[Any, float]: Returns the function's result and the average runtime of the last use_last executions.
    """
    results = []
    times = []

    for _ in range(repeat):
        start_time = time.perf_counter()  # Higher precision timer
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.perf_counter()

        results.append(result)  # Store the result
        times.append(end_time - start_time)  # Store the time

    # Use the average time of the last use_last executions
    average_time = sum(sorted(times)[-use_last:]) / use_last

    # Return the result of the last execution and the average tim
    return results[-1], average_time


def run_with_src(compiler: str, test: Test, qemu: bool = False) -> TestResult:  # lab0
    assert test.expected is not None, f"{test.filename} has no expected output."
    src_file_path = os.path.join(TEMP_DIR, Path(test.filename).with_suffix(".c").name)
    with open(src_file_path, "w") as f:
        f.write(r"""
#include <stdio.h>

int read() {
int x;
scanf("%d", &x);
return x;
}

void write(int x) {
printf("%d\n", x);
}
""")
        f.write(open(test.filename).read())

    executable_file_path = os.path.join(
        TEMP_DIR,
        Path(test.filename).with_suffix(
            ".exe" if sys.platform.startswith("win") else ""
        ).name
    )
    try:
        result = subprocess.run(
            [compiler, src_file_path, "-o", executable_file_path],
            capture_output=True,
            timeout=cfg.timeout
        )
        if result.returncode != 0:  # compile error
            return TestResult(test, result_type=ResultType.COMPILE_ERROR)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)

    try:
        input_str = "\n".join(test.inputs) if test.inputs is not None else None
        if qemu:
            assert RV32_QEMU is not None, "qemu-riscv32 not found."
            result, run_time = measure_execution_time(
                subprocess.run,
                [RV32_QEMU, executable_file_path],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=cfg.timeout
            )
        else:
            result, run_time = measure_execution_time(
                subprocess.run,
                [executable_file_path],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=cfg.timeout
            )
        return TestResult(test, result.stdout.strip().split("\n"), result.returncode, run_time=run_time)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.RUN_TIMEOUT)
        
def run_only_compiler(compiler: str, test: Test) -> TestResult:  # lab1, lab2
    assert test.inputs is None, "Not implemented input for lab1 or lab2"
    try:
        result = subprocess.run(
            [compiler, test.filename], capture_output=True, timeout=cfg.timeout)
        if result.returncode != 0:
            return TestResult(test, result_type=ResultType.COMPILE_ERROR)
        return TestResult(test, result_type=ResultType.ACCEPTED)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)

def run_with_ir(compiler: str, test: Test) -> TestResult:  # lab3
    ir_file_path = os.path.join(TEMP_DIR, Path(test.filename).with_suffix(".zir").name)
    assert os.path.exists(IR_PATH), f"{IR_PATH} not found."
    assert test.expected is not None, f"{test.filename} has no expected output."
    try:
        result = subprocess.run(
            [compiler, test.filename, ir_file_path, '--ir'],  # add --ir flag
            capture_output=True,
            timeout=cfg.timeout)
        if result.returncode != 0:  # compile error
            return TestResult(test, result_type=ResultType.COMPILE_ERROR)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)
        
    try:
        result = subprocess.run(
            [PYTHON, IR_PATH, ir_file_path] + (["--ssa"] if cfg.check_ssa else []),
            input="\n".join(test.inputs) if test.inputs is not None else None,
            capture_output=True,
            text=True,
            timeout=cfg.timeout
        )
        output = result.stdout.split("\n")
        output = [line.strip() for line in output if line.strip() != ""]
        if output:
            # extract run step in the last line "Exit with code {exit_code} within {run_step} steps."
            match = re.search(r"within (\d+) steps", output[-1])
            if match:
                run_step = int(match.group(1))
            else:
                run_step = None
            output = output[:-1]
        else:
            run_step = None
        return TestResult(test, output, result.returncode, run_step=run_step)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.RUN_TIMEOUT)

def run_with_asm(compiler: str, test: Test, qemu: bool = False) -> TestResult:  # lab4
    assert test.expected is not None, f"{test.filename} has no expected output."

    # compile to assembly
    assembly_file_path = os.path.join(TEMP_DIR, Path(test.filename).with_suffix(".s").name)
    try:
        result = subprocess.run(
            [compiler, test.filename, assembly_file_path] + (['--qemu'] if qemu else []),  # add --qemu flag
            capture_output=True,
            timeout=cfg.timeout
        )
        if result.returncode != 0:  # compile error
            return TestResult(test, result_type=ResultType.COMPILE_ERROR)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)
    
    if qemu:
        assert RV32_GCC is not None, "riscv32-unknown-elf-gcc not found."
        assert RV32_QEMU is not None, "qemu-riscv32 not found."

        # compile assembly to executable
        executable_file_path = os.path.join(TEMP_DIR, Path(test.filename).stem)
        try:
            result = subprocess.run(
                [RV32_GCC, assembly_file_path, "-o", executable_file_path] + cfg.extra_cflags,
                capture_output=True,
                timeout=cfg.timeout
            )
            if result.returncode != 0:  # compile error
                return TestResult(test, result_type=ResultType.COMPILE_ERROR)
        except subprocess.TimeoutExpired:
            return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)
        
        # run with qemu
        try:
            result, run_time = measure_execution_time(
                subprocess.run,
                [RV32_QEMU, executable_file_path],
                input="\n".join(test.inputs) if test.inputs is not None else None,
                capture_output=True,
                text=True,
                timeout=cfg.timeout
            )
            return TestResult(test, result.stdout.split(), result.returncode, run_time=run_time)
        except subprocess.TimeoutExpired:
            return TestResult(test, result_type=ResultType.RUN_TIMEOUT)
        
    else:   # run with venus
        assert JAVA is not None, "java not found."
        assert os.path.exists(VENUS_JAR), f"{VENUS_JAR} not found."

        try:
            result = subprocess.run(
                [JAVA, "-jar", VENUS_JAR, assembly_file_path, '-ahs', '-ms', '-1'],         # -ms -1: ignore max step limit
                input="\n".join(test.inputs) if test.inputs is not None else None,
                capture_output=True,
                text=True,
                timeout=cfg.timeout
            )
            result_step = subprocess.run(
                [JAVA, "-jar", VENUS_JAR, assembly_file_path, '-ahs', '-n', '-ms', '-1'],   # -n: only output step count
                input="\n".join(test.inputs) if test.inputs is not None else None,
                capture_output=True,
                text=True,
                timeout=cfg.timeout
            )
            run_step = int(result_step.stdout.strip())
            return TestResult(test, result.stdout.strip().split("\n")[0], result.returncode, concat_output=True, run_step=run_step)
        except subprocess.TimeoutExpired:
            return TestResult(test, result_type=ResultType.RUN_TIMEOUT)

def summary(test_results: list[TestResult]):
    assert len(test_results) > 0, "no tests found."

    def path_to_print(path) -> str:
        path = Path(path)
        if path.is_relative_to(TEST_DIR):
            return path.relative_to(TEST_DIR).as_posix()
        else:
            return path.as_posix()
    
    def colored_result(result: ResultType) -> str:
        color_map = {
            ResultType.ACCEPTED: green,
            ResultType.WRONG_ANSWER: red,
            ResultType.RUN_TIMEOUT: blue,
            ResultType.COMPILE_TIMEOUT: blue,
            ResultType.COMPILE_ERROR: yellow,
            ResultType.RUNTIME_ERROR: yellow
        }
        return color_map[result](result.name.replace("_", " ").title())

    # Create a Rich Table
    table = Table(title="Test Results", box=None)
    table.add_column("Test File", style="bold", justify="left")
    table.add_column("Result", justify="left")
    if test_results[0].run_time is not None:
        table.add_column("Time", justify="right")
    elif test_results[0].run_step is not None:
        table.add_column("Steps", justify="right")

    for test_result in test_results:
        test_file = path_to_print(test_result.test.filename)
        result = colored_result(test_result.result)
        if test_result.run_time is not None:
            table.add_row(test_file, result, f"{test_result.run_time*1000:.2f} ms")
        elif test_result.run_step is not None:
            table.add_row(test_file, result, f"{test_result.run_step} steps")
        else:
            table.add_row(test_file, result)

    print()
    print(table)

    passed = len([test for test in test_results if test.result == ResultType.ACCEPTED])
    print()
    assert passed == len(test_results), f"{passed}/{len(test_results)} tests passed."
    print(green("All tests passed!"))
    print()


def test_lab(source_folder: str, lab: str):
    print(box(f"Running {lab} test..."))

    if lab == 'lab0':
        tests = list((Path(source_folder) / "appends" / "lab0").glob("*.sy"))
    elif 'bonus' in lab:
        tests = list((Path(TEST_DIR) / "tests" / "lab4").glob("*.sy"))
        if lab == 'bonus2':
            tests += list((Path(source_folder) / "appends" / "bonus2").glob("*.sy"))
        elif lab == 'bonus4':
            tests += list((Path(source_folder) / "appends" / "bonus4").glob("*.sy"))
    else:
        tests = list((Path(TEST_DIR) / "tests" / lab).glob("*.sy"))
    tests = sorted(tests)

    test_func = {
        'lab0': partial(run_with_src, qemu=cfg.use_qemu),
        'lab1': run_only_compiler,
        'lab2': run_only_compiler,
        'lab3': run_with_ir,
        'lab4': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus1': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus2': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus3': partial(run_with_asm, qemu=True),
        'bonus4': partial(run_with_asm, qemu=True)
    }[lab]
    
    if lab == 'lab0':
        # test coverage
        try:
            result = subprocess.run([PYTHON, COVERAGE_PATH, *map(str, tests)], check=True)
            print(green(f"lab0 coverage test passed."))
        except:
            raise ValueError(f"lab0 coverage test failed.")

        compiler = CC if not cfg.use_qemu else RV32_GCC
        assert compiler is not None, "gcc or clang not found." if not cfg.use_qemu else "riscv32-unknown-elf-gcc not found."
    else:
        compiler = shutil.which("compiler", path=source_folder)
        assert compiler is not None, "compiler not found in the source folder."

    tests = [Test.parse_file(str(test)) for test in tests]

    if cfg.parallel:
        results = {}
        with ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(test_func, compiler, test): i for i, test in enumerate(tests)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                results[index] = result
        test_results = [results[i] for i in track(range(len(tests)), description="Running tests", disable=not cfg.verbose)]
    else:
        test_results = [test_func(compiler, test) for test in track(tests, description="Running tests", disable=not cfg.verbose)]
    
    summary(test_results)

    if lab == 'lab0':
        assert len(test_results) >= 5, "lab0 test cases are less than 5."
    else:
        assert (Path(source_folder) / "reports" / f"{lab}.pdf").exists(), \
            f"reports/{lab}.pdf not found."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test your compiler.")
    parser.add_argument("lab", type=str, help="Which lab to test",
                            choices=[
                                "lab0", "lab1", "lab2", "lab3", "lab4",
                                "bonus1", "bonus2", "bonus3", "bonus4"
                            ])
    parser.add_argument("repo_path", type=str, nargs='?', default=os.getcwd(), help="Path to your repository. Defaults to the current working directory.")
    args = parser.parse_args()
    repo_path, lab = args.repo_path, args.lab
    
    cfg_path = Path(repo_path) / "config.toml"
    if cfg_path.exists():
        for k, v in toml.load(cfg_path).items():
            if hasattr(cfg, k):
                assert type(getattr(cfg, k)) == type(v), f"Type mismatch for {k}."
                setattr(cfg, k, v)
    try:
        test_lab(repo_path, lab)
    except Exception as e:
        if cfg.verbose:
            print(Traceback())
        print(red(f"Error: {e}"))
        sys.exit(1)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
