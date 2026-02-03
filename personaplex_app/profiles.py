from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class PersonaProfile:
    name: str
    voice_prompt: str
    text_prompt: str


class ProfileError(RuntimeError):
    pass


def load_profiles(profile_path: Path) -> List[PersonaProfile]:
    if not profile_path.exists():
        raise ProfileError(f"Profiles file not found: {profile_path}")

    data = json.loads(profile_path.read_text())
    if not isinstance(data, list):
        raise ProfileError("Profiles file must contain a list of profiles.")

    profiles: List[PersonaProfile] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ProfileError("Each profile must be a JSON object.")
        try:
            profile = PersonaProfile(
                name=str(entry["name"]),
                voice_prompt=str(entry["voice_prompt"]),
                text_prompt=str(entry["text_prompt"]),
            )
        except KeyError as exc:
            raise ProfileError(f"Missing required profile field: {exc}") from exc
        profiles.append(profile)

    if not profiles:
        raise ProfileError("Profiles file is empty. Add at least one profile entry.")

    return profiles


def index_profiles(profiles: List[PersonaProfile]) -> Dict[str, PersonaProfile]:
    return {profile.name.lower(): profile for profile in profiles}


def get_profile(
    profile_name: str,
    profile_path: Path,
) -> PersonaProfile:
    profiles = load_profiles(profile_path)
    index = index_profiles(profiles)
    profile = index.get(profile_name.lower())
    if not profile:
        available = ", ".join(sorted(index.keys()))
        raise ProfileError(
            f"Unknown profile '{profile_name}'. Available: {available}"
        )
    return profile


def resolve_prompt_path(prompt: str, base_dir: Path) -> Path:
    path = Path(prompt)
    if not path.is_absolute():
        path = base_dir / path
    return path
