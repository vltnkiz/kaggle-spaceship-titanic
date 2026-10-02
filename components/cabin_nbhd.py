"""Cabin neighbourhood: who lives within K cabin numbers of you on the same deck and side.

Built over the whole frame (dev + test, no target). Needs Deck/CabinNum/Side (`cabin`) and
NoSpend (`spend`). K is fixed up front, not tuned.

NbhdCount:        other passengers on the same Deck/Side with |CabinNum - own| <= K.
NbhdCryoShare:    share of those neighbours with known CryoSleep who are asleep (NaN if none known).
NbhdNoSpendShare: share of those neighbours with NoSpend == 1 (NaN if no neighbours).
DeckPos:          (CabinNum - min) / (max - min) over the same Deck/Side (NaN if max == min).
Every column, the count included, is NaN when the row's Deck, Side or CabinNum is missing.
"""
import numpy as np

NAME = "cabin_nbhd"
KIND = "feature"

K = 10


def build(frame):
    n = len(frame)
    count, cryo_share, nospend_share, deck_pos = (np.full(n, np.nan) for _ in range(4))

    located = (frame["Deck"].notna() & frame["Side"].notna() & frame["CabinNum"].notna()).to_numpy()
    cryo_known = frame["CryoSleep"].notna().to_numpy(dtype=float)
    cryo_true = (frame["CryoSleep"] == True).to_numpy(dtype=float)  # noqa: E712 - object column
    nospend = (frame["NoSpend"] == 1).to_numpy(dtype=float)
    num = frame["CabinNum"].to_numpy(dtype=float)
    key = (frame["Deck"].astype(str) + "/" + frame["Side"].astype(str)).to_numpy()

    for k in np.unique(key[located]):
        rows = np.flatnonzero(located & (key == k))
        order = rows[np.argsort(num[rows], kind="stable")]
        nums = num[order]
        lo = np.searchsorted(nums, nums - K, side="left")
        hi = np.searchsorted(nums, nums + K, side="right")

        def window_sum(values):
            v = values[order]
            cum = np.concatenate([[0.0], np.cumsum(v)])
            return cum[hi] - cum[lo] - v  # neighbours only, not the row itself

        others = (hi - lo - 1).astype(float)
        known = window_sum(cryo_known)
        count[order] = others
        cryo_share[order] = np.divide(window_sum(cryo_true), known, out=np.full(len(order), np.nan), where=known > 0)
        nospend_share[order] = np.divide(window_sum(nospend), others, out=np.full(len(order), np.nan), where=others > 0)
        span = nums[-1] - nums[0]
        if span > 0:
            deck_pos[order] = (nums - nums[0]) / span

    frame["NbhdCount"] = count
    frame["NbhdCryoShare"] = cryo_share
    frame["NbhdNoSpendShare"] = nospend_share
    frame["DeckPos"] = deck_pos
    return frame
