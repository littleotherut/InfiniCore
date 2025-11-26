import importlib
import logging
import pathlib

from infiniop.ninetoothed.build import BUILD_DIRECTORY_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

CURRENT_FILE_PATH = pathlib.Path(__file__)

SRC_DIR_PATH = CURRENT_FILE_PATH.parent.parent / "src"


def _find_and_build_ops():
    ops_path = SRC_DIR_PATH / "infiniop" / "ops"
    logging.info("Scanning ops directory at %s", ops_path)

    if not ops_path.is_dir():
        logging.error("Ops directory missing: %s", ops_path)
        return

    for op_dir in ops_path.iterdir():
        ninetoothed_path = op_dir / "ninetoothed"
        logging.info("Processing op directory %s", op_dir)

        if ninetoothed_path.is_dir():
            module_path = ninetoothed_path / "build"
            relative_path = module_path.relative_to(SRC_DIR_PATH)
            import_name = ".".join(relative_path.parts)
            logging.info("Importing build module %s", import_name)

            try:
                module = importlib.import_module(import_name)
            except Exception as import_error:
                logging.exception(
                    "Failed to import %s from %s", import_name, module_path
                )
                continue

            if not hasattr(module, "build"):
                logging.error("Module %s misses build()", import_name)
                continue

            logging.info("Running build() for %s", import_name)

            try:
                module.build()
            except Exception as build_error:
                logging.exception("build() failed for %s", import_name)
            else:
                logging.info("build() completed for %s", import_name)
        else:
            logging.info("Skipping %s; no ninetoothed directory", op_dir)


if __name__ == "__main__":
    logging.info("Ensuring build directory %s exists", BUILD_DIRECTORY_PATH)
    BUILD_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

    logging.info("Starting op builds")
    _find_and_build_ops()
    logging.info("Finished op builds")
