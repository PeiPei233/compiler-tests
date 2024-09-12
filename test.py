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

### Settings ###


TEST_DIR = os.path.dirname(__file__)
TIMEOUT = 10
IR_PATH = os.path.join(TEST_DIR, "ir.py")
VENUS_JAR = os.path.join(TEST_DIR, "venus.jar")
COVERAGE_PATH = os.path.join(TEST_DIR, "coverage.py")
PYTHON = sys.executable  # always use the current python
JAVA = shutil.which("java")
CC = shutil.which("gcc") or shutil.which("clang")
RV32_GCC = shutil.which("riscv32-unknown-elf-gcc")
RV32_QEMU = shutil.which("qemu-riscv32")
USE_QEMU = os.environ.get("USE_QEMU", "").lower() in ["1", "true"]
NO_PARALLEL = os.environ.get("NO_PARALLEL", "").lower() in ["1", "true"]

TEMP_DIR = tempfile.mkdtemp()
atexit.register(shutil.rmtree, TEMP_DIR)

### Color Utils ###


def red(s: str) -> str:
    return f"\033[31m{s}\033[0m"

def green(s: str) -> str:
    return f"\033[32m{s}\033[0m"

def blue(s: str) -> str:
    return f"\033[34m{s}\033[0m"

def yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"

def box(s: str) -> str:
    return f"\033[1;7;37m{s}\033[0m"

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
            assert "Input:" in comments[0], f"Error: {filename} has non-paired input/output"
            assert "Output:" in comments[1], f"Error: {filename} has non-paired input/output"
            input = comments[0].replace("Input:", "").split()
            if input[0] == "None":
                input = None
            expected = comments[1].replace("Output:", "").split()
            if expected[0] == "None":
                expected = None
            return Test(filename, input, expected, False)
        else:
            assert False, f"Error: {filename} heading comment is invalid"

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
                assert result_type is None, "Error: run_ret_code and result_type cannot be both not None."
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

def run_with_src(compiler: str, test: Test, qemu: bool = False) -> TestResult:  # lab0
    assert test.expected is not None, f"Error: {test.filename} has no expected output."
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
            timeout=TIMEOUT
        )
        if result.returncode != 0:  # compile error
            return TestResult(test, result_type=ResultType.COMPILE_ERROR)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)

    try:
        input_str = "\n".join(test.inputs) if test.inputs is not None else None
        if qemu:
            assert RV32_QEMU is not None, "Error: qemu-riscv32 not found."
            start_time = time.process_time()
            result = subprocess.run(
                [RV32_QEMU, executable_file_path],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            end_time = time.process_time()
        else:
            start_time = time.process_time()
            result = subprocess.run(
                [executable_file_path],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            end_time = time.process_time()
        return TestResult(test, result.stdout.strip().split("\n"), result.returncode, run_time=end_time - start_time)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.RUN_TIMEOUT)
        
def run_only_compiler(compiler: str, test: Test) -> TestResult:  # lab1, lab2
    assert test.inputs is None, "Not implemented input for lab1 or lab2"
    try:
        result = subprocess.run(
            [compiler, test.filename], capture_output=True, timeout=TIMEOUT)
        if result.returncode != 0:
            return TestResult(test, result_type=ResultType.COMPILE_ERROR)
        return TestResult(test, result_type=ResultType.ACCEPTED)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)

def run_with_ir(compiler: str, test: Test) -> TestResult:  # lab3
    ir_file_path = os.path.join(TEMP_DIR, Path(test.filename).with_suffix(".zir").name)
    assert os.path.exists(IR_PATH), f"Error: {IR_PATH} not found."
    assert test.expected is not None, f"Error: {test.filename} has no expected output."
    try:
        result = subprocess.run(
            [compiler, test.filename, ir_file_path, '--ir'],  # add --ir flag
            capture_output=True,
            timeout=TIMEOUT)
        if result.returncode != 0:  # compile error
            return TestResult(test, result_type=ResultType.COMPILE_ERROR)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)
        
    try:
        result = subprocess.run(
            [PYTHON, IR_PATH, ir_file_path],
            input="\n".join(test.inputs) if test.inputs is not None else None,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        output = result.stdout.split("\n")
        output = [line.strip() for line in output if line.strip() != ""]
        # extract run step in the last line "Exit with code {exit_code} within {run_step} steps."
        match = re.search(r"within (\d+) steps", output[-1])
        assert match is not None, f"Error: {test.filename} has no run step."
        run_step = int(match.group(1))
        output = output[:-1]
        return TestResult(test, output, result.returncode, run_step=run_step)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.RUN_TIMEOUT)

def run_with_asm(compiler: str, test: Test, qemu: bool = False) -> TestResult:  # lab4
    assert test.expected is not None, f"Error: {test.filename} has no expected output."

    # compile to assembly
    assembly_file_path = os.path.join(TEMP_DIR, Path(test.filename).with_suffix(".s").name)
    try:
        result = subprocess.run(
            [compiler, test.filename, assembly_file_path] + (['--qemu'] if qemu else []),  # add --qemu flag
            capture_output=True,
            timeout=TIMEOUT
        )
        if result.returncode != 0:  # compile error
            return TestResult(test, result_type=ResultType.COMPILE_ERROR)
    except subprocess.TimeoutExpired:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)
    
    if qemu:
        assert RV32_GCC is not None, "Error: riscv32-unknown-elf-gcc not found."
        assert RV32_QEMU is not None, "Error: qemu-riscv32 not found."

        # compile assembly to executable
        executable_file_path = os.path.join(TEMP_DIR, Path(test.filename).stem)
        try:
            result = subprocess.run(
                [RV32_GCC, assembly_file_path, "-o", executable_file_path],
                capture_output=True,
                timeout=TIMEOUT
            )
            if result.returncode != 0:  # compile error
                return TestResult(test, result_type=ResultType.COMPILE_ERROR)
        except subprocess.TimeoutExpired:
            return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT)
        
        # run with qemu
        try:
            start_time = time.process_time()
            result = subprocess.run(
                [RV32_QEMU, executable_file_path],
                input="\n".join(test.inputs) if test.inputs is not None else None,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
            end_time = time.process_time()
            return TestResult(test, result.stdout.split(), result.returncode, run_time=end_time - start_time)
        except subprocess.TimeoutExpired:
            return TestResult(test, result_type=ResultType.RUN_TIMEOUT)
        
    else:   # run with venus
        assert JAVA is not None, "Error: java not found."
        assert os.path.exists(VENUS_JAR), f"Error: {VENUS_JAR} not found."

        try:
            result = subprocess.run(
                [JAVA, "-jar", VENUS_JAR, assembly_file_path, '-ms', '-1'],         # -ms -1: ignore max step limit
                input="\n".join(test.inputs) if test.inputs is not None else None,
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            result_step = subprocess.run(
                [JAVA, "-jar", VENUS_JAR, assembly_file_path, '-n', '-ms', '-1'],   # -n: only output step count
                input="\n".join(test.inputs) if test.inputs is not None else None,
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            run_step = int(result_step.stdout.strip())
            return TestResult(test, result.stdout.strip().split("\n")[0], result.returncode, concat_output=True, run_step=run_step)
        except subprocess.TimeoutExpired:
            return TestResult(test, result_type=ResultType.RUN_TIMEOUT)

def summary(test_results: list[TestResult]):
    if len(test_results) == 0:
        print(red("Error: no tests found."))
        return

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

    # get the longest filename
    max_filename = max([len(path_to_print(test_result.test.filename))
                        for test_result in test_results])
    print()
    for test_result in test_results:
        # align the filename
        print(f"{path_to_print(test_result.test.filename).ljust(max_filename)}    ", end="")
        print(colored_result(test_result.result).ljust(25), end="")
        if test_result.run_time is not None:
            print(f"{test_result.run_time*1000:.2f}ms", end="")
        elif test_result.run_step is not None:
            print(f"{test_result.run_step} steps", end="")
        print()
    passed = len([test for test in test_results if test.result == ResultType.ACCEPTED])
    print()
    if passed == len(test_results):
        print(green("All tests passed!"))
    else:
        print(f"{passed}/{len(test_results)} tests passed.")
    print()


def test_lab(source_folder: str, lab: str):
    print(box(f"Running {lab} test..."))

    if lab == 'lab0':
        tests = list((Path(source_folder) / "appends" / "lab0").glob("*.sy"))
    elif 'bonus' in lab:
        tests = list((Path(TEST_DIR) / "tests" / "lab4").glob("*.sy"))
        if lab == 'bonus1':
            tests += list((Path(source_folder) / "appends" / "bonus1").glob("*.sy"))
    else:
        tests = list((Path(TEST_DIR) / "tests" / lab).glob("*.sy"))
    tests = sorted(tests)

    test_func = {
        'lab0': partial(run_with_src, qemu=USE_QEMU),
        'lab1': run_only_compiler,
        'lab2': run_only_compiler,
        'lab3': run_with_ir,
        'lab4': partial(run_with_asm, qemu=USE_QEMU),
        'bonus1': partial(run_with_asm, qemu=USE_QEMU),
        'bonus2': partial(run_with_asm, qemu=USE_QEMU),
        'bonus3': partial(run_with_asm, qemu=True),
        'bonus4': partial(run_with_asm, qemu=True)
    }[lab]
    
    if lab == 'lab0':
        # test coverage
        try:
            result = subprocess.run([PYTHON, COVERAGE_PATH, *map(str, tests)], check=True)
            print(green(f"lab0 coverage test passed."))
        except:
            print(red(f"Error: lab0 coverage test failed."))

        compiler = CC if not USE_QEMU else RV32_GCC
        assert compiler is not None, "Error: gcc or clang not found." if not USE_QEMU else "Error: riscv32-unknown-elf-gcc not found."
    else:
        compiler = shutil.which("compiler", path=source_folder)
        assert compiler is not None, "Error: compiler not found in the source folder."

    tests = [Test.parse_file(str(test)) for test in tests]

    if NO_PARALLEL:
        test_results = [test_func(compiler, test) for test in tests]
    else:
        results = {}
        with ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(test_func, compiler, test): i for i, test in enumerate(tests)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                results[index] = result
        test_results = [results[i] for i in range(len(tests))]
    
    summary(test_results)

    if lab != 'lab0':
        if not (Path(source_folder) / "reports" / f"{lab}.pdf").exists():
            print(red(f"Error: reports/{lab}.pdf not found."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test your compiler.")
    parser.add_argument("lab", type=str, help="Which lab to test",
                            choices=[
                                "lab0", "lab1", "lab2", "lab3", "lab4",
                                "bonus1", "bonus2", "bonus3", "bonus4"
                            ])
    parser.add_argument("source", type=str, help="Path to your repository.")
    args = parser.parse_args()
    source, lab = args.source, args.lab
    test_lab(source, lab)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
