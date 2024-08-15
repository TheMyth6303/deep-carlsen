import matplotlib.pyplot as plt
from IPython import display


def plot(td_losses, episodes):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(td_losses)
    plt.ylim(ymin=0)
    plt.text(len(td_losses)-1, td_losses[-1], str(td_losses[-1]))
    plt.show(block=False)
    plt.pause(.1)
    plt.suptitle(f'{episodes} episodes done')
