import torch
import torch.nn as nn
import torch.optim as optim
import sys
if '../model/' not in sys.path: sys.path.append('../model/')
from models import negative_log_partial_likelihood, getPatientGroupedLoaders, c_index
from custom_dataset import Dataset_DNN_Survival, Dataset_DNN_Classifier, custom_collate_DNN_survival, custom_collate_DNN_classifier
from tqdm import tqdm

class DNN(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features,
                 emb_dim = 640, 
                 dropout_rate=1e-6, 
                 activation=nn.ReLU):
        super(DNN, self).__init__()
        
        self.act = activation()
        self.drop = nn.Dropout(p=dropout_rate)
        self.out_features = out_features
        
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, emb_dim)
        self.fc3 = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.fc_out = nn.Linear(emb_dim, out_features)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc3(x)
        x = self.act(x)
        x = self.drop(x)

        if self.out_features==1: x = self.norm(x)
        
        x = self.fc_out(x)
        return x
    
def train_classifier(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = None):
    best = 0,0
    for epoch in range(num_epochs):
        model.train()
        with tqdm(enumerate(train_loader), total=len(train_loader),desc='TRAINING') as pbar:
            for batch_idx, (data, target) in pbar:
                optimizer.zero_grad()
                output = model(data)
                # assert False
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Epoch':f'{epoch+1}/{num_epochs}, Loss: {loss.item():.4f}'})

            # Evaluation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in tqdm(test_loader,desc='TESTING'):
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            if accuracy > best[0]:
                best = accuracy,epoch
                if saveName is not None:
                    torch.save(model.state_dict(), saveName)
                    print(f'Saved {saveName} at epoch {epoch}')
            print(f'Test Accuracy: {accuracy:.2f}%, ({correct} of {total})')
    print(f'Best Accuracy: {best[0]:.2f}% at epoch {best[1]}')


def train_survival(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = None):
    best = 0,0
    for epoch in range(num_epochs):
        model.train()
        with tqdm(enumerate(train_loader), total=len(train_loader),desc='TRAINING') as pbar:
            for batch_idx, (data, time, event) in pbar:
                optimizer.zero_grad()
                output = model(data)
                survival = torch.stack([time, event], dim=1)
                loss = criterion(survival, output)
                if torch.isnan(loss):
                    print(survival)
                    print(output)
                    assert False
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Epoch':f'{epoch+1}/{num_epochs}, Loss: {loss.item():.4f}'})
                # if batch_idx % 20000 == 0:
                #     print('')

            # Evaluation
            model.eval()
            all_times = []
            all_events = []
            all_risks = []

            with torch.no_grad():
                for data, time, event in tqdm(test_loader,desc='TESTING'):
                    output = model(data)
                    risk = output.squeeze()
                    all_risks.extend(risk.cpu().numpy())
                    all_times.extend(time.cpu().numpy())
                    all_events.extend(event.cpu().numpy())
            survival = np.column_stack((all_times, all_events))
            c_index_value = c_index(np.array(all_risks), survival)
            if c_index_value > best[0]:
                best = c_index_value,epoch
                if saveName is not None:
                    torch.save(model.state_dict(), saveName)
                    print(f'Saved {saveName} at epoch {epoch}')
            print(f'C-Index: {c_index_value:.4f}')
    print(f'Best C-Index: {best[0]:.4f} at epoch {best[1]}')

def run_dnn(class_data,
            df,
            activation_fn = nn.ReLU,
            dropout_rate = 1e-6,
            input_dim = 150,
            survival=False,
            device = 'cuda:0',
            num_epochs = 2,
            n_folds = None,
            test_size = .2
            ):
    
    output_dim = 1 if survival else max(class_data['CANCER_TYPE_INT'].unique()) + 1
    criterion = negative_log_partial_likelihood if survival else nn.CrossEntropyLoss()
    train_fn = train_survival if survival else train_classifier

    to_merge = ['barcode','time','censor', 'patient_id'] if survival else ['barcode','CANCER_TYPE_INT', 'patient_id']

    df_merged = df.merge(
        class_data[to_merge], 
        left_index=True, 
        right_on='barcode'
        )
    
    dataset_class = Dataset_DNN_Survival if survival else Dataset_DNN_Classifier
    dataset = dataset_class(df_merged, device)
    collate_fn = custom_collate_DNN_survival if survival else custom_collate_DNN_classifier
    folds = getPatientGroupedLoaders(dataset, df_merged, n_folds=n_folds, test_size = test_size, batch_size = 500, collate=collate_fn)

    for i, (train_loader, test_loader) in enumerate(folds):
        model = DNN(
            in_features=input_dim,
            out_features=output_dim,
            dropout_rate=dropout_rate,
            activation=activation_fn
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=.0001)
        print(f'\nFOLD {i+1}')
        # if saveName is not None: saveName = './best_models/' + saveName.split('.')[0] + f'_fold{i+1}.pt'
        train_fn(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = None)
