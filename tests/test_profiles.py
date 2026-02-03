import json
from pathlib import Path
import unittest

from personaplex_app.profiles import get_profile, load_profiles


class TestProfiles(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.profile_path = self.repo_root / "personaplex_app" / "profiles.json"

    def test_profiles_json_is_valid(self):
        profiles = load_profiles(self.profile_path)
        self.assertGreater(len(profiles), 0)

    def test_default_profile_exists(self):
        profile = get_profile("assistant", self.profile_path)
        self.assertEqual(profile.name.lower(), "assistant")


if __name__ == "__main__":
    unittest.main()
