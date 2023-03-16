import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import string
import glob
from random_human import random_human
from scipy.stats import t


def plots(xs, ys, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):
    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        plt.plot(x,ys[i], linewidth=1.5,color=color[i],) #linestyle=(0, (i+3, 1, 2*i, 1)),)
    #plt.legend(loc=loc, ncol=1)
    #plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".pdf"))
    plt.close()

def plots_err(xs, ys, ystd, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):

    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        #plt.errorbar(x, ys[i], xerr=0.5, yerr=2*ystd[i], label=legends[i], color=color[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
        plt.plot(x,ys[i], color=color[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
        if True: #i==0:
            plt.fill_between(x, np.array(ys[i])-2*np.array(ystd[i]), np.array(ys[i])+2*np.array(ystd[i]), color=color[i], alpha=0.1)
    #plt.legend(loc=loc, ncol=1)
    #plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".pdf"))
    plt.close()

def find_checkpoint(base_path, postfix):
    model_dicts = []
    game_names = []
    for path_to_load in sorted(glob.glob(base_path + '/*' + postfix), reverse=False):
        for job_lib_file in sorted(glob.glob(path_to_load + '/*' + '_bestq.pkl'), reverse=False):
            model_dict = torch.load(job_lib_file, map_location=torch.device('cpu'))
            model_dicts.append(model_dict)
            game_name = str(os.path.basename(path_to_load))[:-2]
            game_names.append(game_name)
    return model_dicts, game_names

if __name__ == '__main__':
    base_path = '../swin_results/model_savedir/'
    model_dicts1, game_names1 = find_checkpoint(base_path, '00')
    model_dicts2, game_names2 = find_checkpoint(base_path, '01')
    #print(game_names1, game_names2)

    # game_name = 'krull'
    # titile_name = string.capwords(game_name.replace("_", " "))
    # path1 = '../swin_results/model_savedir/' + game_name + '00/'+game_name+'_bestq.pkl'
    # path2 = '../swin_results/model_savedir/' + game_name + '01/'+game_name+'_bestq.pkl'

    # model_dict1 = torch.load(path1, map_location=torch.device('cpu'))
    # model_dict2 = torch.load(path2, map_location=torch.device('cpu'))

    legends = ['Swin DQN', 'Double DQN']
    perf_range = np.arange(0, 8, 0.1)
    perf_scores1 = np.zeros(len(perf_range))
    perf_scores2 = np.zeros(len(perf_range))

    for i, model_dict1 in enumerate(model_dicts1):
        model_dict2 = model_dicts2[i]
        game_name = game_names1[i]
        assert game_name == game_names2[i]

        info = model_dict1['info']
        perf1 = model_dict1['perf']
        perf2 = model_dict2['perf']
        titile_name = string.capwords(game_name.replace("_", " "))

        steps1 = perf1['steps']
        steps2 = perf2['steps']
        eval_steps1 = perf1['eval_steps']
        eval_steps2 = perf2['eval_steps']

        y1_mean_scores = perf1['eval_rewards']
        y1_std_scores = perf1['eval_stds']
        y1q = perf1['q_record']

        y2_mean_scores = perf2['eval_rewards']
        y2_std_scores = perf2['eval_stds']
        y2q = perf2['q_record']


    # ## Mean Eval Normalized
    #     mean_score1 = (y1_mean_scores[-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

    #     mean_score2 = (y2_mean_scores[-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

    #     print(titile_name,'&', round(y2_mean_scores[-1],2), '&', round(y2_std_scores[-1],2),'&', round(mean_score2,2),'&', round(y1_mean_scores[-1],2), '&' , round(y1_std_scores[-1],2), '&', round(mean_score1,2), '\\\\')
    #     print('\\hline')

    ## Highest Eval Normalized
        highest_score1 = (perf1['highest_eval_score'][-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

        highest_score2 = (perf2['highest_eval_score'][-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

        print(game_name, perf1['highest_eval_score'][-1], round(highest_score1,2), perf2['highest_eval_score'][-1], round(highest_score2, 2))

    # ## Performance Profiles
    #     samples1 = np.random.normal(y1_mean_scores[-1], y1_std_scores[-1], 100)
    #     normalized_samples1 = (samples1-random_human[game_name][0])*100/(random_human[game_name][1]-random_human[game_name][0])

    #     samples2 = np.random.normal(y2_mean_scores[-1], y2_std_scores[-1], 100)
    #     normalized_samples2 = (samples2-random_human[game_name][0])*100/(random_human[game_name][1]-random_human[game_name][0])

    #     for x in normalized_samples1:
    #         for i in range(len(perf_range)):
    #             if x >= perf_range[i]*100:
    #                 perf_scores1[i] += 1
    #             else:
    #                 break

    #     for x in normalized_samples2:
    #         for i in range(len(perf_range)):
    #             if x >= perf_range[i]*100:
    #                 perf_scores2[i] += 1
    #             else:
    #                 break



    ### AUC
        # auc1 = 0
        # auc2 = 0
        # auc_dqn = 0
        # for i in range (0, min(len(y1_mean_scores), len(y2_mean_scores)), 2):
        # #for i in range (int(min(len(y1_mean_scores), len(y2_mean_scores))/2), min(len(y1_mean_scores), len(y2_mean_scores))):
        #     auc1 += y1_mean_scores[i]
        #     auc2 += y2_mean_scores[i]
        #     auc_dqn += random_human[game_name][2]
        
        # print(game_name, auc1/abs(auc_dqn), auc2/abs(auc_dqn))

    ## Mean
        title = "Mean Evaluation Scores in "+ titile_name
        plots_err(
            [eval_steps1, eval_steps2],
            [y1_mean_scores, y2_mean_scores],
            [y1_std_scores, y2_std_scores],
            "Steps",
            "Scores",
            title,
            legends,
        )

        # plots(
        #     [eval_steps1, eval_steps2],
        #     [y1_mean_scores, y2_mean_scores],
        #     "Steps",
        #     "Scores",
        #     title,
        #     legends,
        #     loc="upper left"
        # )

        title = "Maximal Q-values in "+ titile_name
        plots(
            [steps1, steps2],
            [y1q, y2q],
            "Steps",
            "Q values",
            title,
            legends,
            loc="upper left"
        )



    # # ### Performance Profiles
    # perf_scores1 = perf_scores1/4900
    # perf_scores2 = perf_scores2/4900
    # #print(perf_scores1)
    # #print(perf_scores2)

    # title = "Performance Profiles"
    # plots(
    #     [perf_range, perf_range],
    #     [perf_scores1, perf_scores2],
    #     "Human Normalized Score (\u03C4)",
    #     "Fraction of Runs with Score > \u03C4",
    #     title,
    #     legends,
    #     loc="upper left"
    # )