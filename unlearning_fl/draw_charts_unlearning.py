from tbparse import SummaryReader
from matplotlib import pyplot as plt
import seaborn as sns
import os

plt.style.use("seaborn-whitegrid")
log_dir = "./unlearning_fl/logging_results_2"
chart_folder = "./unlearning_fl/"
reader = SummaryReader(log_dir, extra_columns={'dir_name'})
df = reader.tensors
print(df)

draw_accuracy = True

if draw_accuracy:
    dfw = df.loc[df['tag'].str.contains('accuracy')]
    # dfw['value'] = dfw['value'].apply(lambda val: val * 100.0)
else:
    dfw = df.loc[df['tag'].str.contains('loss')]

plt.figure(figsize=(8, 6))
configs = [("birds", "blue"), ("aircrafts", "red"), ("cifar100", "orange")]
for cfg in configs:
    dataset = cfg[0]
    color = cfg[1]
    for sanitized in [True, False]:
        plt.subplot(1, 1, 1)

        linestyle = "-" if sanitized else "--"
        label = dataset + " w/o client u" if sanitized else dataset + " with client u"

        string_to_match = dataset + "_sanitized" if sanitized else dataset + "/"
        df = dfw[dfw['dir_name'].str.contains(string_to_match)]
        x = [i for i in range(0, df.shape[0])]
        if df.shape[0] != 0:
            y = df["value"].tolist()

        g = sns.lineplot(x=x, y=y, label=label, linestyle=linestyle, color=color)
        # if draw_accuracy:
        #     print(" max ", current_max)
        #     g.set_ylim([0.0, current_max + 5.0])

        g.set_xlim([0, 100])
        # title =
        # g.set_title(title,
        #             fontsize=16, pad=20)
        if draw_accuracy:
            g.set_ylabel('Accuracy', fontsize=18)
            metric = 'accuracy_comparison'
            g.legend(loc='lower right')
            # g.legend(loc='upper left')
        else:
            g.set_ylabel('Loss', fontsize=18)
            metric = 'loss_comparison'
            g.legend(loc='upper right')

        g.tick_params(axis='both', which='major', labelsize=12)
        g.tick_params(axis='both', which='minor', labelsize=12)
        g.set_xlabel('Round', fontsize=18)
        g.get_legend().set_title(None)
        plt.setp(g.get_legend().get_texts(), fontsize='13')

        # k_string = "_K" + str(k) if k == 100 else ""
        filename = "mit-b0_sanitized_vs_original.pdf"
        g.get_figure().savefig(os.path.join(chart_folder, filename),
                               format='png', bbox_inches='tight')
        plt.show()
