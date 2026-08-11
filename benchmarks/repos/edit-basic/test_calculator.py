import unittest

from calculator import answer


class CalculatorTest(unittest.TestCase):
    def test_answer(self):
        self.assertEqual(answer(), 2)


if __name__ == "__main__":
    unittest.main()
