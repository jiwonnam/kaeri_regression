import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
train_features = pd.read_csv('train_features.csv')
train_target = pd.read_csv('train_target.csv', index_col='id')
test_features = pd.read_csv('test_features.csv')


def plot_show(idx, df, title="", save=False):
    f, axes = plt.subplots(4, 1, figsize=(6, 9))
    # f.tight_layout()
    # plt.subplots_adjust(top=0.88)

    for i in range(1, 5):
        print(df[df['id'] == idx]['S'+str(i)].values)
        axes[i-1].plot(df[df['id'] == idx]['S'+str(i)].values)
        axes[i-1].set_title('S{}, {}'.format(i, title))
        axes[i-1].set_xlabel('time')
    if save:
        plt.savefig("S_plots_{}_{}.png".format(idx, title))
    else:
        plt.show()


def training_data_distribution(train_target, save=False):
    plt.figure()
    sns.distplot(train_target['X'], kde=False, bins=18)
    plt.title("X value distribution")
    if save:
        plt.savefig("X distribution.png")
    else:
        plt.show()

    plt.figure()
    sns.distplot(train_target['Y'], kde=False, bins=18)
    plt.title("Y value distribution")
    if save:
        plt.savefig("Y distribution.png")
    else:
        plt.show()

    plt.figure()
    sns.scatterplot(train_target['X'], train_target['Y'])
    plt.title("Position distribution")
    if save:
        plt.savefig("X and Y distribution.png")
    else:
        plt.show()

    plt.figure()
    sns.distplot(train_target['M'], kde=False)
    plt.title("Mass distribution")
    if save:
        plt.savefig("M distribution.png")
    else:
        plt.show()

    plt.figure()
    sns.distplot(train_target['V'], kde=False)
    plt.title("V distribution")
    if save:
        plt.savefig("V distribution.png")
    else:
        plt.show()

    plt.figure()
    sns.scatterplot(train_target['M'], train_target['V'])
    plt.title("M and V distribution")
    if save:
        plt.savefig("M and V distribution.png")
    else:
        plt.show()


def plot_fft_preprocessing():
    import numpy as np
    fs = 5
    # sampling frequency
    fmax = 25
    # sampling period
    dt = 1/fs
    # length of signal
    N = 75

    df = fmax/N
    f = np.arange(0,N)*df

    xf = np.fft.fft(train_features[train_features.id==0]['S2'].values)*dt
    print(len(np.abs(xf[0:int(N/2+1)])))

    plt.plot(f[0:int(N/2+1)],np.abs(xf[0:int(N/2+1)]))
    plt.xlabel('frequency(Hz)');
    plt.ylabel('abs(xf)');
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass
    # training_data_distribution(train_target, save=True)

    # See corner cases
    # plot_show(1011, train_features, title="400,0", save=True)

    # training_data_distribution(train_target)
    # Check relative position of Sensors S1, S2, S3, and S4
    # plot_count = 3
    # train_target = pd.read_csv('train_target.csv')
    # x_y_0_0_df = train_target[(train_target['X'] == 0) & (train_target['Y'] == 300.0)]['id'].values
    # for i in range(len(x_y_0_0_df)):
    #     title = "X == 0, Y == 0"
    #     plot_show(x_y_0_0_df[i], train_features, title)

    # It looks like S1 and S2 are apart from the center in the same distance.
    # If x is the same, still S1 and S2 go together. This means they are symmetric in y-axis
    # x_y_0_not0_df = train_target[(train_target['X'] == 0) & (train_target['Y'] != 0)]['id'].values
    # for i in range(plot_count):
    #     title = "X == 0, Y != 0"
    #     plot_show(x_y_0_not0_df[i], train_features, title)

    # x_y_not0_0_df = train_target[(train_target['X'] != 0) & (train_target['Y'] == 0)]['id'].values
    # for i in range(plot_count):
    #     title = "X != 0, Y == 0"
    #     plot_show(x_y_not0_0_df[i], train_features, title)


    # m_175_df = train_target[train_target['M']==175]['id'].values
    # for i in range(len(m_175_df)):
    #     plot_show(m_175_df[i], train_features)
    #
    # m_25_df = train_target[train_target['M']==25]['id'].values
