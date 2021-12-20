import unittest
from download_data import check_hash_correct, get_sha256
import os


class TestHash(unittest.TestCase):
    def test_sha256(self):
        test_dir = os.path.dirname(__file__)
        file = os.path.join(test_dir, 'hash_sample/hash.txt')
        expected_hash = 'a119e9fd61ee902dbd10c8058a7474ea50cf7ae7edfd17fbf400871b0b31e628'
        self.assertEqual(get_sha256(file), expected_hash)
        return

    def test_checkhash(self):
        test_dir = os.path.dirname(__file__)
        file = os.path.join(test_dir, 'hash_sample/hash.txt')
        expected_hash = 'a119e9fd61ee902dbd10c8058a7474ea50cf7ae7edfd17fbf400871b0b31e628'
        self.assertTrue(check_hash_correct(file, expected_hash))
        self.assertFalse(check_hash_correct(file, 'chicken'))
        return
