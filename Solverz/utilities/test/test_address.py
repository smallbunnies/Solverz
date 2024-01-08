from Solverz.utilities.address import Address, combine_Address


def test_address():
    a = Address()
    a.add('x')

    a.add('y', 2)
    assert a['y'] == slice(0, 2, None)
    a.update('x', 3)
    assert a['x'] == slice(0, 3, None)

    try:
        a.update('z', 3)
    except KeyError as k:
        assert k.args[0] == "Non-existent name z, use Address.add() instead"

    a.update('x', 0)
    try:
        a['x']
    except IndexError as i:
        assert i.args[0] == "index 0 is out of bounds for axis 0 with size 0"

    a.update('x', 5)
    a0 = a.derive_alias('0')
    assert a0['x0'] == slice(0, 5, None)
    assert a0.size['x0'] == 5
    assert a0.size['y0'] == 2
    assert a0.total_size == 7

    a2 = combine_Address(a, a0)
    assert a2['y0'] == slice(12, 14, None)

    b = Address()
    b.add('x')
    b.add('y')
    b.update('x', 5)
    b.update('y', 2)

    assert a != a0
    assert a == b
