from __future__ import annotations

import pandas as pd


class Model:
    name: str = "base"

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
