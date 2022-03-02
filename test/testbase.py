from unittest import TestCase

class TestBase(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.DATA_PATH = "test/testdata/bosz_5000_test.h5"

        self.params = {"DATA_PATH": self.DATA_PATH
                        "wStart": 71.0,
        }


    