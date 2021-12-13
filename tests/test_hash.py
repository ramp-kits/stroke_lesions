import unittest
import hash_check


class TestHash(unittest.TestCase):
    def test_sha256(self):
        file = 'tests/hash_sample/hash.txt'
        expected_hash = 'a119e9fd61ee902dbd10c8058a7474ea50cf7ae7edfd17fbf400871b0b31e628'
        self.assertEqual(hash_check.get_sha256(file), expected_hash)
        return

    def test_checkhash(self):
        file = 'tests/hash_sample/hash.txt'
        expected_hash = 'a119e9fd61ee902dbd10c8058a7474ea50cf7ae7edfd17fbf400871b0b31e628'
        self.assertTrue(hash_check.check_hash_correct(file, expected_hash))
        self.assertFalse(hash_check.check_hash_correct(file, 'chicken'))
        return