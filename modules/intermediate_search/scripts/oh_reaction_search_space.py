from dataclasses import dataclass


@dataclass(frozen=True)
class AdsorbateState:
    name: str
    oxygen_count: int
    hydrogen_count: int


ADSORBATE_LIBRARY = {
    "bare": AdsorbateState("bare", 0, 0),
    "O": AdsorbateState("O", 1, 0),
    "OH": AdsorbateState("OH", 1, 1),
    "H2O": AdsorbateState("H2O", 1, 2),
    "O2": AdsorbateState("O2", 2, 0),
    "OOH": AdsorbateState("OOH", 2, 1),
    "H2O2": AdsorbateState("H2O2", 2, 2),
}


def apply_delta(state: AdsorbateState, delta_o: int = 0, delta_h: int = 0) -> AdsorbateState | None:
    target_o = state.oxygen_count + delta_o
    target_h = state.hydrogen_count + delta_h
    for candidate in ADSORBATE_LIBRARY.values():
        if candidate.oxygen_count == target_o and candidate.hydrogen_count == target_h:
            return candidate
    return None


def allowed_neighbors(state_name: str) -> list[tuple[str, str]]:
    state = ADSORBATE_LIBRARY[state_name]
    moves = [
        ("+O", apply_delta(state, delta_o=1)),
        ("-O", apply_delta(state, delta_o=-1)),
        ("+H", apply_delta(state, delta_h=1)),
        ("-H", apply_delta(state, delta_h=-1)),
    ]
    return [(move, nxt.name) for move, nxt in moves if nxt is not None]


if __name__ == "__main__":
    for name in ADSORBATE_LIBRARY:
        print(name, "->", allowed_neighbors(name))
