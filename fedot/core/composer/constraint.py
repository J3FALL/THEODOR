from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_validation import validate


def constraint_function(chain: Chain):
    try:
        validate(chain)
        return True
    except ValueError:
        return False
