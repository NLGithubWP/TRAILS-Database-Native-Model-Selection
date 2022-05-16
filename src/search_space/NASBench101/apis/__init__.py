from search_space.NASBench101.apis import nb101_api


def get_nasbench101_api(bench_path):
    nb101_data = nb101_api.NASBench(bench_path)
    return {"api": nb101_api, "nb101_data": nb101_data}
