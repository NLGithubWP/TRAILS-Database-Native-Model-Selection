
import orjson
from pkg2.pkg2 import pgg2_test
from pkg1 import pgg1_test


def print_hell(a: str):
    b = pgg1_test() + pgg2_test()
    return orjson.dumps({"data": b, "types": a}).decode('utf-8')

