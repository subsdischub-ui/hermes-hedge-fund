"""Strategy #2 R&D — research-only signal candidates + an evaluation harness.

ISOLATED from the live agent: signals here are NOT registered in the live skill
library. They are evaluated against 9y history (CPCV/DSR net of fees + an
orthogonality gate vs directional momentum). Only survivors, after review +
explicit approval, graduate to live A/B.
"""
