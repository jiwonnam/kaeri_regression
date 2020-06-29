import pandas as pd


def merge_submit(xy_submit, m_submit, v_submit, merged_output):
    merged = pd.read_csv('sample_submission.csv')

    xy = pd.read_csv(xy_submit)
    m = pd.read_csv(m_submit)
    v = pd.read_csv(v_submit)

    assert len(xy) == len(m) == len(v)

    merged.iloc[:, 1] = xy.iloc[:, 1]
    merged.iloc[:, 2] = xy.iloc[:, 2]
    merged.iloc[:, 3] = m.iloc[:, 3]
    merged.iloc[:, 4] = v.iloc[:, 4]

    merged.to_csv(merged_output, index=False)


if __name__ == "__main__":
    pass
    # test_xy = "result/submit_xy.csv"
    # test_m = "result/submit_m.csv"
    # test_v = "result/submit_v.csv"
    # merged_output = "result/test_merged.csv"
    # merge_submit(test_xy, test_m, test_v, merged_output)