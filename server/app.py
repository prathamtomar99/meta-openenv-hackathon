import uvicorn

from environment.server import app


def main() -> None:
	uvicorn.run("environment.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
	main()
