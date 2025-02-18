import atexit
import signal
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
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, update_wrapper
from typing import Any, Callable, TypeVar
from difflib import unified_diff

import toml
from rich import print
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Group, RenderableType

from rich.traceback import install
install()

### Settings ###


@dataclass
class EnvironmentConfig:
    test_dir: str = os.path.dirname(__file__)
    temp_dir: str = tempfile.mkdtemp()
    output_dir: str = temp_dir
    
    zero_ir_path: str = os.path.join(test_dir, "ir", "zero.py")
    accipit_ir_path: str = os.path.join(test_dir, "ir", "accipit.py")
    venus_jar: str = os.path.join(test_dir, "venus.jar")
    coverage_py: str = os.path.join(test_dir, "coverage.py")
    io_c: str = os.path.join(test_dir, "libs", "io.c")
    timer_c_path: str = os.path.join(test_dir, "libs", "timer.c")
    
    python: str = sys.executable
    java: str | None = shutil.which("java")
    cc: str | None = shutil.which("gcc") or shutil.which("clang")
    rv32_gcc: str | None = shutil.which("riscv32-unknown-elf-gcc")
    rv32_qemu: str | None = shutil.which("qemu-riscv32")

    def __post_init__(self):
        atexit.register(shutil.rmtree, self.temp_dir)

envs = EnvironmentConfig()

test_score = 0

@dataclass
class TestConfig:
    verbose: bool = False
    use_qemu: bool = False
    use_accipit: bool = False
    parallel: bool = True
    check_ssa: bool = False
    precise_timing: bool = True
    timeout: float = 10.0
    extra_cflags: list[str] = field(default_factory=lambda: [envs.io_c])
    
cfg = TestConfig()

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
        comments: list[str] = []
        for line in content:
            # get comment, start with //
            if line.strip().startswith("//"):
                comments.append(line.strip()[2:].strip())
            else:
                break
        if len(comments) >= 2 and comments[0].startswith("Input:") and comments[1].startswith("Output:"):   # input and output
            input = comments[0].replace("Input:", "").split()
            if input[0] == "None":
                input = []
            expected = comments[1].replace("Output:", "").split()
            if expected[0] == "None":
                expected = []
            return Test(filename, input, expected, False)
        elif len(comments) >= 1 and "Error" in comments[0]:   # should fail
            return Test(filename, None, None, True)
        else:   # should success
            return Test(filename, None, None, False)

    def __str__(self):
        return f"Test({self.filename}, {self.inputs}, {self.expected}, {self.should_fail})"

class ResultType(Enum):
    ACCEPTED = auto()
    WRONG_ANSWER = auto()
    RUN_TIMEOUT = auto()
    COMPILE_TIMEOUT = auto()
    COMPILE_ERROR = auto()
    RUNTIME_ERROR = auto()
    
class TimeType(Enum):
    CYCLES = auto()
    STEPS = auto()
    NANOSECONDS = auto()

class TestResult:
    def __init__(self,
                 test: Test,
                 output: str | None | list[str] = None,
                 result_type: ResultType | None = None,
                 run_time: float | None = None,
                 run_time_type: TimeType = TimeType.STEPS,
                 error: Exception | None = None
    ) -> None:
        self.test = test
        self.output = output
        self.run_time = run_time
        self.run_time_type = run_time_type
        self.error = error
        if test.should_fail:
            self.result = ResultType.ACCEPTED if result_type == ResultType.COMPILE_ERROR else ResultType.WRONG_ANSWER
        else:
            if result_type is not None and result_type != ResultType.ACCEPTED:
                self.result = result_type
            else:
                if test.expected is None:
                    self.result = ResultType.WRONG_ANSWER if output else ResultType.ACCEPTED
                else:
                    self.result = ResultType.ACCEPTED if output == test.expected else ResultType.WRONG_ANSWER

_T = TypeVar("_T")
def execute_with_timing(func: Callable[..., _T], *args: Any, repeat: int = 10, use_last: int = 5, **kwargs: Any) -> tuple[list[_T], list[float]]:
    """
    Run a function multiple times and return the results and times.

    Args:
        func (Callable): The function to measure (no parameters).
        repeat (int): Total number of repetitions, default is 10.
        use_last (int): Number of last results to return, default is 5.

    Returns:
        Tuple[List[Any], List[float]]: List of results and list of times.
    """
    results: list[_T] = []
    times: list[float] = []

    for _ in range(repeat):
        start_time = time.perf_counter()  # Higher precision timer
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.perf_counter()

        results.append(result)  # Store the result
        times.append(end_time - start_time)  # Store the time

    return results[-use_last:], times[-use_last:]

def run_commands(commands: list[list[str]], *args: Any, **kwargs: Any) -> None:
    for command in commands:
        subprocess.run(command, capture_output=True, check=True, timeout=cfg.timeout, *args, **kwargs)

def test_exception_handling(func: Callable[..., TestResult]) -> Callable[..., TestResult]:
    def wrapper(compiler: str, test: Test, *args: Any, **kwargs: Any) -> TestResult:
        try:
            return func(compiler, test, *args, **kwargs)
        # except AssertionError:
        #     raise
        except subprocess.TimeoutExpired as e:
            if e.cmd[0] == compiler:
                return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT, error=e)
            else:
                return TestResult(test, result_type=ResultType.RUN_TIMEOUT, error=e)
        except subprocess.CalledProcessError as e:
            if e.cmd[0] == compiler:
                return TestResult(test, result_type=ResultType.COMPILE_ERROR, error=e)
            else:
                return TestResult(test, result_type=ResultType.RUNTIME_ERROR, error=e)
        # except Exception as e:
        #     return TestResult(test, result_type=ResultType.RUNTIME_ERROR, error=e)
    return update_wrapper(wrapper, func)

@test_exception_handling
def compile_run_result(compiler: str, test: Test, src_file_path: str, use_qemu: bool) -> TestResult:
    executable_file_path = os.path.join(
        envs.output_dir,
        Path(test.filename).with_suffix(
            ".exe" if sys.platform.startswith("win") else ""
        ).name
    )
    if cfg.precise_timing:
        run_commands([
            [compiler, src_file_path, envs.timer_c_path, "-o", executable_file_path, "-Wno-implicit-function-declaration"] + \
                (["-DRV32ASM"] if use_qemu else []) + cfg.extra_cflags,
        ])
    else:
        run_commands([
            [compiler, src_file_path, "-o", executable_file_path] + cfg.extra_cflags,
        ])

    if use_qemu:
        assert envs.rv32_qemu is not None, "qemu-riscv32 not found."
        cmd = [envs.rv32_qemu, executable_file_path]
    else:
        cmd = [executable_file_path]
 
    results, run_times = execute_with_timing(
        subprocess.run,
        cmd,
        input="\n".join(test.inputs) if test.inputs is not None else None,
        capture_output=True,
        text=True,
        timeout=cfg.timeout,
        check=True,
    )
    sample_result = results[-1]
    sample_output = sample_result.stdout.strip().split("\n")
    # case 1: Execution time: %llu cycles
    match = re.search(r"Execution time: (\d+) cycles", sample_output[-1])
    if match:
        cycles: list[int] = []
        for result in results:
            output = result.stdout.strip().split("\n")
            time_line = output[-1]
            match = re.search(r"Execution time: (\d+) cycles", time_line)
            assert match, "Execution time not found."
            cycles.append(int(match.group(1)))
        return TestResult(test, sample_output[:-1], run_time=sum(cycles) / len(cycles), run_time_type=TimeType.CYCLES)
    # case 2: Execution time: %ld s %ld ns
    match = re.search(r"Execution time: (\d+) s (\d+) ns", sample_output[-1])
    if match:
        times: list[int] = []
        for result in results:
            output = result.stdout.strip().split("\n")
            time_line = output[-1]
            match = re.search(r"Execution time: (\d+) s (\d+) ns", time_line)
            assert match, "Execution time not found."
            times.append(int(int(match.group(1)) * 1e9 + int(match.group(2))))
        return TestResult(test, sample_output[:-1], run_time=sum(times) / len(times), run_time_type=TimeType.NANOSECONDS)
    # case 3: no match, use run_times avg
    return TestResult(test, sample_output, run_time=int(sum(run_times) / len(run_times) * 1e9), run_time_type=TimeType.NANOSECONDS)

@test_exception_handling
def run_with_src(compiler: str, test: Test, qemu: bool = False) -> TestResult:  # lab0
    assert test.expected is not None, f"{test.filename} has no expected output."
    src_file_path = os.path.join(envs.output_dir, Path(test.filename).with_suffix(".c").name)
    shutil.copy(test.filename, src_file_path)
    
    return compile_run_result(compiler, test, src_file_path, qemu)

@test_exception_handling
def run_only_compiler(compiler: str, test: Test) -> TestResult:  # lab1, lab2
    assert test.inputs is None, "Not implemented input for lab1 or lab2"
    subprocess.run([compiler, test.filename], capture_output=True, timeout=cfg.timeout, check=True)
    return TestResult(test, result_type=ResultType.ACCEPTED)

@test_exception_handling
def run_with_ir(compiler: str, test: Test, accipit: bool) -> TestResult:  # lab3
    ir_file_path = os.path.join(
        envs.output_dir,
        Path(test.filename).with_suffix(
            ".acc" if accipit else ".zir"
        ).name
    )
    ir_path = envs.accipit_ir_path if accipit else envs.zero_ir_path
    assert os.path.exists(ir_path), f"{ir_path} not found."
    assert test.expected is not None, f"{test.filename} has no expected output."
    result = subprocess.run(
        [compiler, test.filename, ir_file_path, '--ir'],  # add --ir flag
        capture_output=True,
        check=True,
        timeout=cfg.timeout)
    
    result = subprocess.run(
        [envs.python, ir_path, ir_file_path] + (["--ssa"] if cfg.check_ssa and not accipit else []),
        input="\n".join(test.inputs) if test.inputs is not None else None,
        capture_output=True,
        text=True,
        timeout=cfg.timeout,
        check=True
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
    return TestResult(test, output, run_time=run_step, run_time_type=TimeType.STEPS)

@test_exception_handling
def run_with_asm(compiler: str, test: Test, qemu: bool = False) -> TestResult:  # lab4
    assert test.expected is not None, f"{test.filename} has no expected output."

    # compile to assembly
    assembly_file_path = os.path.join(envs.output_dir, Path(test.filename).with_suffix(".S").name)
    subprocess.run(
        [compiler, test.filename, assembly_file_path] + (['--venus'] if not qemu else []),  # add --venus flag
        capture_output=True,
        timeout=cfg.timeout,
        check=True
    )
    
    if qemu:
        assert envs.rv32_gcc is not None, "riscv32-unknown-elf-gcc not found."
        assert envs.rv32_qemu is not None, "qemu-riscv32 not found."

        return compile_run_result(envs.rv32_gcc, test, assembly_file_path, use_qemu=True)    
    else:   # run with venus
        assert envs.java is not None, "java not found."
        assert os.path.exists(envs.venus_jar), f"{envs.venus_jar} not found."

        result = subprocess.run(
            [envs.java, "-jar", envs.venus_jar, assembly_file_path, '-ahs', '-ms', '-1'],         # -ms -1: ignore max step limit
            input="\n".join(test.inputs) if test.inputs is not None else None,
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
            check=True
        )
        output = result.stdout.strip().split("\n")
        # if the last line is "Exited with error code {exit_code}", remove it
        if len(output) > 1 and output[-1].startswith("E"):
            output = output[:-1]
        output = [line.strip() for line in output if line.strip() != ""]
        result_step = subprocess.run(
            [envs.java, "-jar", envs.venus_jar, assembly_file_path, '-ahs', '-n', '-ms', '-1'],   # -n: only output step count
            input="\n".join(test.inputs) if test.inputs is not None else None,
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
            check=True
        )
        run_step = int(result_step.stdout.strip())
        return TestResult(test, output, run_time=run_step, run_time_type=TimeType.STEPS)

def summary(test_results: list[TestResult], source_folder: str):
    assert len(test_results) > 0, "no tests found."

    def path_to_print(path_str: str) -> str:
        path: Path = Path(path_str)
        if path.is_relative_to(envs.test_dir):
            return path.relative_to(envs.test_dir).as_posix()
        elif path.is_relative_to(Path(source_folder)):
            return path.relative_to(Path(source_folder)).as_posix()
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
    
    def format_time(run_time: float, run_time_type: TimeType) -> str:
        if run_time_type == TimeType.CYCLES:
            return f"{run_time:.2f} cycles"
        elif run_time_type == TimeType.STEPS:
            return f"{run_time} steps"
        elif run_time_type == TimeType.NANOSECONDS:
            if run_time < 1e3:
                return f"{run_time:.2f} ns"
            elif run_time < 1e6:
                return f"{run_time/1e3:.2f} us"
            elif run_time < 1e9:
                return f"{run_time/1e6:.2f} ms"
            else:
                return f"{run_time/1e9:.2f} s"
        else:
            raise ValueError("Invalid TimeType.")
        
    def truncate_lines(s: str, show_lines: int = 10) -> str:
        lines = s.split("\n")
        if len(lines) <= show_lines * 3:
            return s
        omitted = len(lines) - show_lines * 2
        return "\n".join(
            lines[:show_lines] + \
            [f"[dim]... {omitted} lines omitted ...[/dim]"] + \
            lines[-show_lines:]
        )

    if cfg.verbose:
        for test_result in test_results:
            # Create a panel for each test
            test_filename = path_to_print(test_result.test.filename)
            result_text = colored_result(test_result.result)

            title = f"[bold]{test_filename}[/bold] {result_text}"
            # WRONG_ANSWER case
            if test_result.result == ResultType.WRONG_ANSWER:
                body = []

                if test_result.test.should_fail:
                    body.append("[yellow]Expected to compile with error, but compiled successfully.[/yellow]")
                    print()
                    print(Panel(Group(*body), title=title, title_align='left', expand=True))
                else:
                    expected = test_result.test.expected or []
                    got = test_result.output or []

                    # Generate diff
                    diff = unified_diff(
                        expected,
                        got,
                        fromfile="Expected",
                        tofile="Got",
                        lineterm=""
                    )

                    # Highlight diff using Syntax
                    diff_text = "\n".join(diff)
                    if diff_text.strip():
                        diff_syntax = Syntax(truncate_lines(diff_text), "diff", line_numbers=False)
                    else:
                        diff_syntax = "[dim]No differences found[/dim]"

                    body.append(diff_syntax)
                    print()
                    print(Panel(Group(*body), title=title, title_align='left', expand=True, highlight=True))

            # Other non-ACCEPTED errors
            elif test_result.result != ResultType.ACCEPTED:
                body: list[RenderableType] = []

                if isinstance(test_result.error, subprocess.CalledProcessError):
                    try:
                        stdout = test_result.error.stdout.decode().strip() or "[dim]No stdout[/dim]"
                    except:
                        stdout = test_result.error.stdout or "[dim]No stdout[/dim]"
                    try:
                        stderr = test_result.error.stderr.decode().strip() or "[dim]No stderr[/dim]"
                    except:
                        stderr = test_result.error.stderr or "[dim]No stderr[/dim]"
                        
                    returncode = test_result.error.returncode
                    if returncode and returncode < 0:
                        try:
                            return_text = f"{str(signal.Signals(-returncode))} ({returncode})"
                        except ValueError:
                            return_text = f"{returncode}"
                    else:
                        return_text = f"{returncode}"

                    # Add stdout/stderr in a table
                    table = Table(show_header=False, box=None, highlight=True)
                    table.add_column(style="yellow italic", justify="right")
                    table.add_column()
                    table.add_row("command", str(test_result.error.cmd))
                    table.add_row("exit code", return_text)
                    table.add_row("stdout", truncate_lines(stdout))
                    table.add_row("stderr", truncate_lines(stderr))
                    print()
                    print(Panel(table, title=title, title_align='left', expand=True, highlight=True))
                elif isinstance(test_result.error, subprocess.TimeoutExpired):
                    try:
                        assert test_result.error.stdout is not None
                        stdout = test_result.error.stdout.decode().strip() or "[dim]No stdout[/dim]"
                    except:
                        stdout = "[dim]No stdout[/dim]"
                    try:
                        assert test_result.error.stderr is not None
                        stderr = test_result.error.stderr.decode().strip() or "[dim]No stderr[/dim]"
                    except:
                        stderr = "[dim]No stderr[/dim]"
                    table = Table(show_header=False, box=None, highlight=True)
                    table.add_column(style="yellow italic", justify="right")
                    table.add_column()
                    table.add_row("command", str(test_result.error.cmd))
                    table.add_row("Timeout", f"{test_result.error.timeout} seconds")
                    table.add_row("stdout", truncate_lines(stdout))
                    table.add_row("stderr", truncate_lines(stderr))
                    print()
                    print(Panel(table, title=title, title_align='left', expand=True, highlight=True))
                else:
                    body.append("[bold]Error Details:[/bold]")
                    body.append(f"{test_result.error}")
                    print()
                    print(Panel(Group(*body), title=title, title_align='left', expand=True, highlight=True))

    # Create a Rich Table
    table = Table(title="Test Results", box=None, expand=True, highlight=True)
    table.add_column("Test File", style="bold", justify="left")
    table.add_column("Result", justify="left")
    table.add_column("Time", justify="left")

    for test_result in test_results:
        test_file = path_to_print(test_result.test.filename)
        result = colored_result(test_result.result)
        if test_result.run_time is not None:
            table.add_row(test_file, result, format_time(test_result.run_time, test_result.run_time_type))
        else:
            table.add_row(test_file, result)

    print()
    print(table)

    passed = len([test for test in test_results if test.result == ResultType.ACCEPTED])
    print()
    global test_score
    test_score = 100 * passed / len(test_results)
    assert passed == len(test_results), f"{passed}/{len(test_results)} tests passed."
    print(green("All tests passed!"))
    print()

def init_worker(envs: EnvironmentConfig, cfg: TestConfig) -> None:
    globals()['envs'] = envs
    globals()['cfg'] = cfg

def test_lab(source_folder: str, lab: str, files: list[str]) -> None:
    print(box(f"Running {lab} test..."))

    if files:
        tests = [path for file in files for path in (Path(file).rglob("*.sy") if os.path.isdir(file) else [Path(file)])]
    else:
        if lab == 'lab0':
            tests = list((Path(source_folder) / "appends" / "lab0").glob("*.sy"))
        elif 'bonus' in lab:
            tests = list((Path(envs.test_dir) / "tests" / "lab4").glob("*.sy"))
            if lab == 'bonus2':
                tests += list((Path(source_folder) / "appends" / "bonus2").glob("*.sy"))
            elif lab == 'bonus3':
                tests += list((Path(envs.test_dir) / "tests" / "bonus3").glob("*.sy"))
                tests += list((Path(source_folder) / "appends" / "bonus3").glob("*.sy"))
        else:
            tests = list((Path(envs.test_dir) / "tests" / lab).glob("*.sy"))
    tests = sorted(tests)
    
    if lab == 'lab0':
        compiler = envs.cc if not cfg.use_qemu else envs.rv32_gcc
        assert compiler is not None, "gcc or clang not found." if not cfg.use_qemu else "riscv32-unknown-elf-gcc not found."
    else:
        compiler = shutil.which("compiler", path=source_folder)
        assert compiler is not None, "compiler not found in the source folder."

    test_func: dict[str, Callable[[str, Test], TestResult]] = {
        'lab0': partial(run_with_src, qemu=cfg.use_qemu),
        'lab1': run_only_compiler,
        'lab2': run_only_compiler,
        'lab3': partial(run_with_ir, accipit=cfg.use_accipit),
        'lab4': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus1': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus2': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus3': partial(run_with_asm, qemu=True),
    }
    
    tests = [Test.parse_file(str(test)) for test in tests]

    if cfg.parallel:
        results: dict[int, TestResult] = {}
        with ProcessPoolExecutor(initializer=init_worker, initargs=(envs, cfg)) as executor:
            future_to_index = {executor.submit(test_func[lab], compiler, test): i for i, test in enumerate(tests)}
            for future in track(as_completed(future_to_index), total=len(tests), description="Running tests", disable=not cfg.verbose):
                index = future_to_index[future]
                result = future.result()
                results[index] = result
        test_results = [results[i] for i in range(len(tests))]
    else:
        test_results = [test_func[lab](compiler, test) for test in track(tests, description="Running tests", disable=not cfg.verbose)]
    
    summary(test_results, source_folder)

    if not files and lab == 'lab0':
        global test_score
        test_score = 0
        # test coverage
        try:
            result = subprocess.run([envs.python, envs.coverage_py, *map(lambda x: x.filename, tests)], check=True)
            print(green(f"lab0 coverage test passed."))
        except:
            assert False, f"lab0 coverage test failed."
        test_score = min(100, 100 * len(tests) / 5)
        assert len(tests) >= 5, "lab0 test cases are less than 5."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test your compiler.")
    parser.add_argument("lab", type=str, help="Which lab to test",
                            choices=[
                                "lab0", "lab1", "lab2", "lab3", "lab4",
                                "bonus1", "bonus2", "bonus3"
                            ])
    parser.add_argument("repo_path", type=str, nargs='?', default=os.getcwd(), help="Path to your repository. Defaults to the current working directory.")
    parser.add_argument("-f", "--file", nargs="*", help="Specific test files/dirs to run. Will not record score.")
    parser.add_argument("-o", "--output", type=str, help="Output directory for test results.")
    
    str2bool: Callable[[str], bool] = lambda x: x.lower() in {'true', 't', 'yes', 'y', '1'}
    parser.add_argument("-v", "--verbose", type=str2bool, nargs='?', const=True, help="Print detailed error messages.")
    parser.add_argument("--use-qemu", "--use_qemu", "--qemu", type=str2bool, nargs='?', const=True, help="Use qemu-riscv32 to run tests.")
    parser.add_argument("--use-accipit", "--use_accipit", "--accipit", type=str2bool, nargs='?', const=True, help="Use Accipit IR instead of Zero IR to run tests.")
    parser.add_argument("--parallel", type=str2bool, nargs='?', const=True, help="Run tests in parallel.")
    parser.add_argument("--check-ssa", "--check_ssa", type=str2bool, nargs='?', const=True, help="Check SSA form in lab3. (Zero IR Only)")
    parser.add_argument("--precise-timing", "--precise_timing", type=str2bool, nargs='?', const=True, help="Measure precise timing (build with measure_time.c).")
    parser.add_argument("--timeout", type=float, help="Timeout for each test case.")
    parser.add_argument("--extra-cflags", "--extra_cflags", nargs='*', help="Extra cflags for gcc/clang.")
    
    args = parser.parse_args()
    repo_path, lab, files = args.repo_path, args.lab, args.file
    
    if args.output:
        envs.output_dir = args.output
        Path(envs.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"All compiled files will be stored in {envs.output_dir}")
    if args.extra_cflags:
        args.extra_cflags = [flag for flags in args.extra_cflags for flag in flags.split()]
    
    cfg_path = Path(repo_path) / "config.toml"
    if cfg_path.exists():
        for k, v in toml.load(cfg_path).items():
            if hasattr(cfg, k):
                assert type(getattr(cfg, k)) == type(v), f"Type mismatch for {k}. Expected {type(getattr(cfg, k))}, got {type(v)}."
                setattr(cfg, k, v)
    
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            assert type(getattr(cfg, k)) == type(v), f"Type mismatch for {k}. Expected {type(getattr(cfg, k))}, got {type(v)}."
            setattr(cfg, k, v)
    
    if cfg.verbose:     # print configurations
        table = Table(show_header=False, box=None, highlight=True)
        table.add_column(justify="right", style="yellow italic")
        table.add_column()
        for k in sorted(vars(envs)):
            table.add_row(k, str(getattr(envs, k)))
        print(Panel(table, title="Environment Configurations", title_align='left', expand=True, highlight=True, border_style="blue"))
        table = Table(show_header=False, box=None, highlight=True)
        table.add_column(justify="right", style="yellow italic")
        table.add_column()
        for k in sorted(vars(cfg)):
            table.add_row(k, str(getattr(cfg, k)))
        print()
        print(Panel(table, title="Test Configurations", title_align='left', expand=True, highlight=True, border_style="blue"))
    
    failed = False
    try:
        test_lab(repo_path, lab, files)
    except AssertionError as e:
        print(red(f"Error: {e}"))
        failed = True
    if not files:
        if lab == 'lab0':
            print(f"[bold]Test score: {test_score:.2f}[/bold]")
        else:
            if not (Path(repo_path) / "reports" / f"{lab}.pdf").exists():
                print(red(f"Error: reports/{lab}.pdf not found."))
                failed = True
            print(f"[bold]Test score: {test_score:.2f}[/bold]")
    print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %Z"))
    
    sys.exit(1 if failed else 0)
