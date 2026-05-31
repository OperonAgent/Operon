"""
Operon Hermes Planner Renderer — V2.

Renders the agent's reasoning scratchpad in the Operon Engine V1.0.0 style:
  • Rounded corners ┌ ┐ └ ┘
  • Emoji-prefixed labels:    
  • Single-row per field with intelligent truncation
"""


class HermesPlannerRenderer:

    def render(self, scratchpad: dict, theme) -> None:
        """
        Print the scratchpad frame to stdout.
        Accepts the dict from the model's "scratchpad" key.
        """
        obj   = str(scratchpad.get("objective",      "—")).strip()
        wvars = scratchpad.get("workspace_vars",     {})
        draft = str(scratchpad.get("code_draft",     "")).strip()
        nxt   = str(scratchpad.get("next_step",      "—")).strip()

        # Format vars dict
        if isinstance(wvars, dict):
            vars_str = ", ".join(f"{k}={v}" for k, v in wvars.items()) if wvars else "—"
        else:
            vars_str = str(wvars) if wvars else "—"

        # Inline code draft
        if draft:
            draft_str = draft.replace("\n", " ↵ ")
        else:
            draft_str = "—"

        rows = [
            ("", "OBJECTIVE ", obj),
            ("", "VARIABLES ", vars_str),
            ("", "SANDBOX   ", draft_str),
            ("", "NEXT STEP ", nxt),
        ]

        box = theme.planner_box(rows)
        print(f"\n{box}\n")
