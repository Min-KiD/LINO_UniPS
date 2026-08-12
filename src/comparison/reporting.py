"""Pure console-reporting helpers for long comparison inference runs."""


def format_clock_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_optional_gib(value: int | None) -> str:
    if value is None:
        return "unavailable"
    if type(value) is not int or value < 0:
        raise ValueError("memory byte count must be a non-negative integer or null")
    return f"{value / (1024 ** 3):.2f} GiB"


def should_report_progress(
    completed: int,
    total: int,
    interval: int = 100,
) -> bool:
    if total <= 0 or completed <= 0 or completed > total:
        raise ValueError("progress counts must satisfy 1 <= completed <= total")
    if interval <= 0:
        raise ValueError("progress interval must be positive")
    return completed == 1 or completed == total or completed % interval == 0


def estimate_eta_seconds(elapsed: float, completed: int, total: int) -> float:
    if total <= 0 or completed <= 0 or completed > total:
        raise ValueError("ETA counts must satisfy 1 <= completed <= total")
    elapsed = max(0.0, float(elapsed))
    return (elapsed / completed) * (total - completed)
