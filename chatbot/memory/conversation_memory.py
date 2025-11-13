from typing import List


# =============================
# Conversation Memory
# =============================
class Conversation:
    def __init__(self, system_prompt: str):
        self.base_system_prompt = system_prompt
        self.current_style_modifier = ""
        self.messages: List[dict] = [{"role": "system", "content": self._get_full_system_prompt()}]

    def _get_full_system_prompt(self) -> str:
        """Combine base prompt with current style modifier."""
        if self.current_style_modifier:
            return f"{self.base_system_prompt}\n\nCURRENT STYLE OVERRIDE: {self.current_style_modifier}"
        return self.base_system_prompt

    def _update_system_prompt(self):
        """Update the system message with current style."""
        self.messages[0] = {"role": "system", "content": self._get_full_system_prompt()}

    def _detect_style_commands(self, user_text: str) -> bool:
        """Detect and apply style change commands. Returns True if command was processed."""
        text_lower = user_text.lower().strip()
        
        style_commands = {
            "speak more formally": "Use formal, professional language with detailed explanations and proper grammar.",
            "be more formal": "Use formal, professional language with detailed explanations and proper grammar.",
            "be more casual": "Use casual, informal language with contractions and conversational style.",
            "speak more casually": "Use casual, informal language with contractions and conversational style.",
            "be more technical": "Include technical details, terminology, and in-depth explanations.",
            "explain like i'm 5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "explain like im 5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "eli5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "be more concise": "Give very brief, to-the-point responses with minimal elaboration.",
            "be more detailed": "Provide comprehensive explanations with examples and context.",
            "be more creative": "Use creative language, metaphors, and imaginative explanations.",
            "be more professional": "Use business-appropriate language and maintain professional demeanor.",
            "reset style": "",  # Empty string resets to default
            "default style": "",
            "normal style": ""
        }
        
        for command, modifier in style_commands.items():
            if command in text_lower:
                self.current_style_modifier = modifier
                self._update_system_prompt()
                return True
        
        return False

    def add_user(self, text: str):
        # Check for style commands before adding to conversation
        is_style_command = self._detect_style_commands(text)
        
        if is_style_command:
            # Add a confirmation message for the style change
            style_name = "default" if not self.current_style_modifier else "updated"
            self.messages.append({"role": "user", "content": text})
            self.messages.append({"role": "assistant", "content": f"Got it! I've switched to {style_name} response style."})
        else:
            self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def history(self) -> List[dict]:
        return list(self.messages)

    def get_current_style(self) -> str:
        """Get current style description for debugging."""
        return self.current_style_modifier or "Default conversational style"