import torch
import matplotlib.pyplot as plt


def plot_attribute(stats, variable, attri = 'success_rate'):
    x = [s[variable] for s in stats['dend']]
    y = [s[attri] for s in stats['dend']]
    plt.plot(x, y, label = 'dendrite forward')

    x = [s[variable] for s in stats['receptor']]
    y = [s[attri] for s in stats['receptor']]
    plt.plot(x, y, label = 'receptor forward')

    plt.legend()
    plt.ylabel(attri)
    plt.xlabel(variable)
    plt.title(f'{attri} VS {variable}\n --timeout 4 --batch_size 10 --sequence_length 20 --max_workers 10')
    plt.savefig(f'{variable}_{attri}.png')
    plt.clf()
    
variable = 'n_queried'
stats = torch.load(f'speed_test_{variable}.pt')
plot_attribute(stats, variable, 'success_rate')
plot_attribute(stats, variable, 'avg_seconds')
plot_attribute(stats, variable, 'avd_upload')
plot_attribute(stats, variable, 'avg_download')

variable = 'n_tasks'
stats = torch.load(f'speed_test_{variable}.pt')
plot_attribute(stats, variable, 'success_rate')
plot_attribute(stats, variable, 'avg_seconds')
plot_attribute(stats, variable, 'avd_upload')
plot_attribute(stats, variable, 'avg_download')