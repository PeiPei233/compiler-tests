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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable, TypeVar
from difflib import unified_diff

import toml
from rich import print
from rich.table import Table
from rich.progress import track
from rich.traceback import Traceback
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Group, RenderableType

### Settings ###


@dataclass
class EnvironmentConfig:
    test_dir: str = os.path.dirname(__file__)
    temp_dir: str = tempfile.mkdtemp()
    output_dir: str = temp_dir
    
    ir_path: str = os.path.join(test_dir, "ir.py")
    venus_jar: str = os.path.join(test_dir, "venus.jar")
    coverage_py: str = os.path.join(test_dir, "coverage.py")
    io_c: str = os.path.join(test_dir, "libs", "io.c")
    measure_time_c_path: str = os.path.join(test_dir, "libs", "measure_time.c")
    
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
    parallel: bool = True
    check_ssa: bool = False
    precise_timing: bool = True
    timeout: int = 10
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
def execute_with_timing(func: Callable[..., _T], *args, repeat: int = 10, use_last: int = 5, **kwargs) -> tuple[list[_T], list[float]]: # type: ignore
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

def run_commands(commands: list[list[str]], *args, **kwargs) -> None: # type: ignore
    for command in commands:
        subprocess.run(command, capture_output=True, check=True, timeout=cfg.timeout, *args, **kwargs) # type: ignore

def compile_run_result(compiler: str, src_file_path: str, test: Test, use_qemu: bool, wrap_main: bool) -> TestResult:
    
    executable_file_path = os.path.join(
        envs.output_dir,
        Path(test.filename).with_suffix(
            ".exe" if sys.platform.startswith("win") else ""
        ).name
    )
    object_file_path = os.path.join(
        envs.output_dir,
        Path(test.filename).with_suffix(".o").name
    )
    try:
        if cfg.precise_timing:
            if wrap_main:
                run_commands([
                    [compiler, src_file_path, envs.measure_time_c_path, "-o", executable_file_path, "-Wl,--wrap=main", "-DWRAP_MAIN"] + \
                        (["-DRV32ASM"] if use_qemu else []) + cfg.extra_cflags,
                ])
            else:
                run_commands([
                    [compiler, "-c", src_file_path, "-o", object_file_path, "-Dmain=_orig_main", "-Wno-implicit-function-declaration"],
                    [compiler, object_file_path, envs.measure_time_c_path, "-o", executable_file_path] + \
                        (["-DRV32ASM"] if use_qemu else []) + cfg.extra_cflags,
                ])
        else:
            run_commands([
                [compiler, src_file_path, "-o", executable_file_path] + cfg.extra_cflags,
            ])
    except subprocess.TimeoutExpired as e:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT, error=e)
    except Exception as e:
        return TestResult(test, result_type=ResultType.COMPILE_ERROR, error=e)

    input_str = "\n".join(test.inputs) if test.inputs is not None else None
    
    if use_qemu:
        assert envs.rv32_qemu is not None, "qemu-riscv32 not found."
        cmd = [envs.rv32_qemu, executable_file_path]
    else:
        cmd = [executable_file_path]
    try:
        results, run_times = execute_with_timing(
            subprocess.run,
            cmd,
            input=input_str,
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
    except subprocess.TimeoutExpired as e:
        return TestResult(test, result_type=ResultType.RUN_TIMEOUT, error=e)
    except Exception as e:
        return TestResult(test, result_type=ResultType.RUNTIME_ERROR, error=e)

def run_with_src(compiler: str, test: Test, qemu: bool = False, is_clang: bool = False) -> TestResult:  # lab0
    assert test.expected is not None, f"{test.filename} has no expected output."
    src_file_path = os.path.join(envs.output_dir, Path(test.filename).with_suffix(".c").name)
    shutil.copy(test.filename, src_file_path)
    
    return compile_run_result(compiler, src_file_path, test, qemu, wrap_main=not is_clang)
        
def run_only_compiler(compiler: str, test: Test) -> TestResult:  # lab1, lab2
    assert test.inputs is None, "Not implemented input for lab1 or lab2"
    try:
        subprocess.run([compiler, test.filename], capture_output=True, timeout=cfg.timeout, check=True)
        return TestResult(test, result_type=ResultType.ACCEPTED)
    except subprocess.TimeoutExpired as e:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT, error=e)
    except Exception as e:
        return TestResult(test, result_type=ResultType.COMPILE_ERROR, error=e)

def run_with_ir(compiler: str, test: Test) -> TestResult:  # lab3
    ir_file_path = os.path.join(envs.output_dir, Path(test.filename).with_suffix(".zir").name)
    assert os.path.exists(envs.ir_path), f"{envs.ir_path} not found."
    assert test.expected is not None, f"{test.filename} has no expected output."
    try:
        result = subprocess.run(
            [compiler, test.filename, ir_file_path, '--ir'],  # add --ir flag
            capture_output=True,
            check=True,
            timeout=cfg.timeout)
    except subprocess.TimeoutExpired as e:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT, error=e)
    except Exception as e:
        return TestResult(test, result_type=ResultType.COMPILE_ERROR, error=e)
        
    try:
        result = subprocess.run(
            [envs.python, envs.ir_path, ir_file_path] + (["--ssa"] if cfg.check_ssa else []),
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
    except subprocess.TimeoutExpired as e:
        return TestResult(test, result_type=ResultType.RUN_TIMEOUT, error=e)
    except Exception as e:
        return TestResult(test, result_type=ResultType.RUNTIME_ERROR, error=e)

def run_with_asm(compiler: str, test: Test, qemu: bool = False) -> TestResult:  # lab4
    assert test.expected is not None, f"{test.filename} has no expected output."

    # compile to assembly
    assembly_file_path = os.path.join(envs.output_dir, Path(test.filename).with_suffix(".S").name)
    try:
        subprocess.run(
            [compiler, test.filename, assembly_file_path] + (['--qemu'] if qemu else []),  # add --qemu flag
            capture_output=True,
            timeout=cfg.timeout,
            check=True
        )
    except subprocess.TimeoutExpired as e:
        return TestResult(test, result_type=ResultType.COMPILE_TIMEOUT, error=e)
    except Exception as e:
        return TestResult(test, result_type=ResultType.COMPILE_ERROR, error=e)
    
    if qemu:
        assert envs.rv32_gcc is not None, "riscv32-unknown-elf-gcc not found."
        assert envs.rv32_qemu is not None, "qemu-riscv32 not found."

        return compile_run_result(envs.rv32_gcc, assembly_file_path, test, use_qemu=True, wrap_main=True)    
    else:   # run with venus
        assert envs.java is not None, "java not found."
        assert os.path.exists(envs.venus_jar), f"{envs.venus_jar} not found."

        try:
            result = subprocess.run(
                [envs.java, "-jar", envs.venus_jar, assembly_file_path, '-ahs', '-ms', '-1'],         # -ms -1: ignore max step limit
                input="\n".join(test.inputs) if test.inputs is not None else None,
                capture_output=True,
                text=True,
                timeout=cfg.timeout,
                check=True
            )
            output = result.stdout.strip().split("\n")[:-1]  # remove the last line "Exit with code {exit_code} within {run_step} steps."
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
        except subprocess.TimeoutExpired as e:
            return TestResult(test, result_type=ResultType.RUN_TIMEOUT, error=e)
        except Exception as e:
            return TestResult(test, result_type=ResultType.RUNTIME_ERROR, error=e)

def summary(test_results: list[TestResult], source_folder: str):
    assert len(test_results) > 0, "no tests found."

    def path_to_print(path: str) -> str: # type: ignore
        path: Path = Path(path)
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
                        diff_syntax = Syntax(diff_text, "diff", line_numbers=False)
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
                    table.add_row("stdout", stdout)
                    table.add_row("stderr", stderr)
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
                    table.add_row("stdout", stdout)
                    table.add_row("stderr", stderr)
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
        def check_is_clang(compiler: str) -> bool:
            result = subprocess.run([compiler, "--version"], capture_output=True, text=True)
            return "clang" in result.stdout
        is_clang = check_is_clang(compiler)
    else:
        compiler = shutil.which("compiler", path=source_folder)
        assert compiler is not None, "compiler not found in the source folder."
        is_clang = False

    test_func: dict[str, Callable[[str, Test], TestResult]] = {
        'lab0': partial(run_with_src, qemu=cfg.use_qemu, is_clang=is_clang),
        'lab1': run_only_compiler,
        'lab2': run_only_compiler,
        'lab3': run_with_ir,
        'lab4': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus1': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus2': partial(run_with_asm, qemu=cfg.use_qemu),
        'bonus3': partial(run_with_asm, qemu=True),
    }
    
    tests = [Test.parse_file(str(test)) for test in tests]

    if cfg.parallel:
        results: dict[int, TestResult] = {}
        with ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(test_func[lab], compiler, test): i for i, test in enumerate(tests)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                results[index] = result
        test_results = [results[i] for i in track(range(len(tests)), description="Running tests", disable=not cfg.verbose)]
    else:
        test_results = [test_func[lab](compiler, test) for test in track(tests, description="Running tests", disable=not cfg.verbose)]
    
    summary(test_results, source_folder)

    if not files and lab == 'lab0':
        # test coverage
        try:
            result = subprocess.run([envs.python, envs.coverage_py, *map(lambda x: x.filename, tests)], check=True)
            print(green(f"lab0 coverage test passed."))
        except:
            assert False, f"lab0 coverage test failed."
        global test_score
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
    parser.add_argument("--parallel", type=str2bool, nargs='?', const=True, help="Run tests in parallel.")
    parser.add_argument("--check-ssa", "--check_ssa", type=str2bool, nargs='?', const=True, help="Check SSA form in lab3.")
    parser.add_argument("--precise-timing", "--precise_timing", type=str2bool, nargs='?', const=True, help="Measure precise timing (build with measure_time.c).")
    parser.add_argument("--timeout", type=int, help="Timeout for each test case.")
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
                assert type(getattr(cfg, k)) == type(v), f"Type mismatch for {k}."
                setattr(cfg, k, v)
    
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            assert type(getattr(cfg, k)) == type(v), f"Type mismatch for {k}."
            setattr(cfg, k, v)
    
    if cfg.verbose:
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
    except Exception as e:
        if cfg.verbose and not isinstance(e, AssertionError):
            print(Traceback())
        print(red(f"Error: {e}"))
        failed = True
    if not files:
        if lab == 'lab0':
            print(f"[bold]Expected score: {test_score:.2f}[/bold]")
        else:
            if not (Path(repo_path) / "reports" / f"{lab}.pdf").exists():
                print(red(f"Error: reports/{lab}.pdf not found."))
                failed = True
                report_score = 0
            else:
                report_score = 100
            print(f"[bold]Expected score: {test_score*0.9+report_score*0.1:.2f}[/bold]")
    print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %Z"))
    
    sys.exit(1 if failed else 0)
