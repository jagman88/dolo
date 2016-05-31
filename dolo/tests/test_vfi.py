def test_vfi():

    from dolo import yaml_import
    from dolo.algos.dtmscc.value_iteration import solve_policy

    model = yaml_import("examples/models/rbc_dtmscc.yaml")
    drv = solve_policy(model)


if __name__ == '__main__':

    test_vfi()
