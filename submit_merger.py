import pandas as pd


def merge_xy_m_v_submits(xy_submit, m_submit, v_submit, merged_output):
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


def ensemble_submits(outfile, *files):
    weight = 1.0/(len(files))  # Equal weight
    result = pd.read_csv("sample_submission.csv")
    for file in files:
        r = pd.read_csv(file)
        result.iloc[:, 1:] += weight * r.iloc[:, 1:]

    result.to_csv(outfile, index=False)


if __name__ == "__main__":
    pass
