# utils.py

import torch
from matplotlib import pyplot as plt

def save_checkpoint(file_path,model,optimizer,best_acc,num_epochs):
    torch.save({
      'arch': 'vision_transformer',
      'num_epochs': num_epochs,
      'best_acc': best_acc,
      'model_state_dict': model.state_dict(),
      'optim_state_dict': optimizer.state_dict(),
    },file_path)
    print('Epochs {} \t Best Accuracy {:.4f}'.format(num_epochs,best_acc))
    print('!!!!!!!!!Checkpoint Saved!!!!!!!')
    
    
def load_checkpoint(file_path,model,optimizer):
    print(file_path)
    checkpoint = torch.load(file_path)
    print('Epochs {} \n Best Accuracy {}'.format(checkpoint['num_epochs'], checkpoint['best_acc']))

    if checkpoint['arch'] == 'vision_transformer':
        model = VisionTransformer()
        for param in model.parameters():
            param.requires_grad= False    
    else: 
        print('Architecture mistmatch')
        return None

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    return model


def plot_summary(train_loss, test_loss):
    # plot test and train loss
    plt.figure(figsize=(20,5))
    plt.plot(train_loss, 'b', label='train')
    plt.plot(test_loss, 'r', label='test')
    plt.legend(loc=1, prop={'size': 30})
    plt.title('Loss Summary')
    plt.show()