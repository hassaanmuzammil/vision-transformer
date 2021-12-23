from data import *
from train import *
from config import *
from utils import *
from model import *

if __name__ == "__main__":
    
    model = VisionTransformer().to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
    
    if pre_trained:
        model.load_state_dict('./logs/default/best_checkpoint.pth')

    model, train_loss, test_loss = train_model(model, loss, optimizer, scheduler, 'best_checkpoint.pth', num_epochs=num_epochs, device=device)
    
    plot_summary(train_loss, test_loss)