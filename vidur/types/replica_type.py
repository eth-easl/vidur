from vidur.types.base_int_enum import BaseIntEnum


class ReplicaType(BaseIntEnum):
    PROMPT = 0
    TOKEN = 1
    MIXED = 2