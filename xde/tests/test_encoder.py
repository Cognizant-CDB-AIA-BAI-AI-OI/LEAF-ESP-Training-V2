import unittest

from xde.encoder import EnumStrictEncoder


class TestEnumStrictEncoder(unittest.TestCase):

    def test_encode(self):
        encoder = EnumStrictEncoder()
        possible_values = ["one", "two", "three", "four", "five"]
        target = "one"
        one_hot = encoder.encode(target, possible_values)
        self.assertEquals(5, len(one_hot), msg="There are 5 values")
        self.assertEquals([1, 0, 0, 0, 0], one_hot)
        target = "three"
        one_hot = encoder.encode(target, possible_values)
        self.assertEquals(5, len(one_hot), msg="There are 5 values")
        self.assertEquals([0, 0, 1, 0, 0], one_hot)
        target = "five"
        one_hot = encoder.encode(target, possible_values)
        self.assertEquals(5, len(one_hot), msg="There are 5 values")
        self.assertEquals([0, 0, 0, 0, 1], one_hot)
        target = "fOUr"
        one_hot = encoder.encode(target, possible_values)
        self.assertEquals(5, len(one_hot), msg="There are 5 values")
        self.assertEquals([0, 0, 0, 1, 0], one_hot, "Encoding of mixed case values failed")

    def test_decode(self):
        encoder = EnumStrictEncoder()
        possible_values = ["one", "two", "three", "four", "five"]
        # One
        encoded = [1, 0, 0, 0, 0]
        decoded = encoder.decode(encoded, possible_values)
        self.assertEquals("one", decoded)
        # Three
        encoded = [0, 0, 1, 0, 0]
        decoded = encoder.decode(encoded, possible_values)
        self.assertEquals("three", decoded)
        # Five
        encoded = [0, 0, 0, 0, 1]
        decoded = encoder.decode(encoded, possible_values)
        self.assertEquals("five", decoded)
