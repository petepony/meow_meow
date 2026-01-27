{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "initial_id",
   "metadata": {
    "collapsed": true,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:32:35.801837Z",
     "start_time": "2026-01-27T10:32:30.933471Z"
    }
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn.functional as F\n",
    "import torch.nn as nn"
   ]
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [
    "class ResidualBlock(nn.Module):\n",
    "    def __init__(self, dim, dropout):\n",
    "        super().__init__()\n",
    "        self.ln = nn.LayerNorm(dim)\n",
    "        self.fc = nn.Linear(dim, dim)\n",
    "        self.gelu = nn.GELU()\n",
    "        self.dropout = nn.Dropout(dropout)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        residual = x\n",
    "        out = self.ln(x)\n",
    "        out = self.fc(out)\n",
    "        out = self.gelu(out)\n",
    "        out = self.dropout(out)\n",
    "        return x + out\n",
    "    \n",
    "class MLP(nn.Module):\n",
    "    def __init__(self, input_dim, hidden_dim, num_blocks, dropout=0.1):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.init_layer = nn.Linear(input_dim, hidden_dim)\n",
    "        self.blocks = nn.ModuleList([\n",
    "            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)\n",
    "        ])\n",
    "        \n",
    "        self.output_layer = nn.Linear(hidden_dim, 1)\n",
    "        self.softplus = nn.Softplus()\n",
    "        \n",
    "    def forward(self, x):\n",
    "        x = self.init_layer(x)\n",
    "        for block in self.blocks:\n",
    "            x = block(x)\n",
    "        \n",
    "        x = self.output_layer(x)\n",
    "        return self.softplus(x).flatten()\n",
    "    \n",
    "    @staticmethod\n",
    "    def criterion(pred, y):\n",
    "        eps = 1e-9\n",
    "        return torch.sqrt(torch.mean(torch.square((y - pred) / (y + eps))))"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:32:35.816841Z",
     "start_time": "2026-01-27T10:32:35.803836Z"
    }
   },
   "id": "2e364f37f2c898e0",
   "execution_count": 2
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "data": {
      "text/plain": "(428910, 9)"
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('../baseline2.csv', index_col=False).iloc[:, 1:]\n",
    "df.shape"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:34:01.143098Z",
     "start_time": "2026-01-27T10:34:00.839780Z"
    }
   },
   "id": "ddc8d1859c9fe1f",
   "execution_count": 17
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "data": {
      "text/plain": "   stock_id  time_id     sigma    target  size    spread  time_diff  \\\n0         0        5  4.499364  4.135767  3179  7.922559         15   \n1         0       11  1.204431  1.444587  1287  4.118409         23   \n2         0       16  2.368527  2.168189  2161  6.476585         35   \n3         0       31  2.573832  2.195261  1962  7.627233         28   \n4         0       62  1.894499  1.747216  1791  4.302926         25   \n\n   log_time_diff  timefunc  log_size  \n0       2.708050  0.310975  8.064322  \n1       3.135494  0.723982  7.160069  \n2       3.555348  0.971598  7.678326  \n3       3.332205  0.849035  7.581720  \n4       3.218876  0.779443  7.490529  ",
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>stock_id</th>\n      <th>time_id</th>\n      <th>sigma</th>\n      <th>target</th>\n      <th>size</th>\n      <th>spread</th>\n      <th>time_diff</th>\n      <th>log_time_diff</th>\n      <th>timefunc</th>\n      <th>log_size</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0</td>\n      <td>5</td>\n      <td>4.499364</td>\n      <td>4.135767</td>\n      <td>3179</td>\n      <td>7.922559</td>\n      <td>15</td>\n      <td>2.708050</td>\n      <td>0.310975</td>\n      <td>8.064322</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>0</td>\n      <td>11</td>\n      <td>1.204431</td>\n      <td>1.444587</td>\n      <td>1287</td>\n      <td>4.118409</td>\n      <td>23</td>\n      <td>3.135494</td>\n      <td>0.723982</td>\n      <td>7.160069</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>0</td>\n      <td>16</td>\n      <td>2.368527</td>\n      <td>2.168189</td>\n      <td>2161</td>\n      <td>6.476585</td>\n      <td>35</td>\n      <td>3.555348</td>\n      <td>0.971598</td>\n      <td>7.678326</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>0</td>\n      <td>31</td>\n      <td>2.573832</td>\n      <td>2.195261</td>\n      <td>1962</td>\n      <td>7.627233</td>\n      <td>28</td>\n      <td>3.332205</td>\n      <td>0.849035</td>\n      <td>7.581720</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>0</td>\n      <td>62</td>\n      <td>1.894499</td>\n      <td>1.747216</td>\n      <td>1791</td>\n      <td>4.302926</td>\n      <td>25</td>\n      <td>3.218876</td>\n      <td>0.779443</td>\n      <td>7.490529</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['log_size'] = np.log(df['size'])\n",
    "df['target'] = df['target'] * 1000\n",
    "df['sigma'] = df['sigma'] * 1000\n",
    "df.head()"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:34:01.173096Z",
     "start_time": "2026-01-27T10:34:01.145097Z"
    }
   },
   "id": "d83769cac448d247",
   "execution_count": 18
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [
    "X = df.loc[:, ['sigma', 'spread', 'log_size', 'timefunc', 'log_time_diff']]\n",
    "y = df['target']\n",
    "\n",
    "X = X.to_numpy()\n",
    "y = y.to_numpy()\n",
    "\n",
    "X = torch.tensor(X, dtype=torch.float32)\n",
    "y = torch.tensor(y, dtype=torch.float32)"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:34:01.794658Z",
     "start_time": "2026-01-27T10:34:01.771655Z"
    }
   },
   "id": "d71d06dea9ed7fde",
   "execution_count": 19
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, shuffle=True, random_state=1)"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:32:36.555958Z",
     "start_time": "2026-01-27T10:32:36.286409Z"
    }
   },
   "id": "c7b172a4bb1134f1",
   "execution_count": 6
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [
    "from torch.utils.data import TensorDataset, DataLoader\n",
    "\n",
    "dataset = TensorDataset(X_train, y_train)\n",
    "dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, pin_memory=True)\n",
    "\n",
    "val_dataset = TensorDataset(X_val, y_val)\n",
    "val_dataloader = DataLoader(val_dataset, batch_size=2048, pin_memory=True)"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:32:36.570960Z",
     "start_time": "2026-01-27T10:32:36.557959Z"
    }
   },
   "id": "8c8266761c3af2dd",
   "execution_count": 7
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [
    "INPUT_DIM = X_train.shape[1]\n",
    "NUM_BLOCKS = 3\n",
    "HIDDEN_DIM = 128\n",
    "DROPOUT_RATE = 0.05\n",
    "\n",
    "model = MLP(\n",
    "    input_dim=INPUT_DIM,\n",
    "    hidden_dim=HIDDEN_DIM,\n",
    "    num_blocks=NUM_BLOCKS,\n",
    "    dropout=DROPOUT_RATE\n",
    ")"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:32:36.600959Z",
     "start_time": "2026-01-27T10:32:36.572959Z"
    }
   },
   "id": "497660a77464f9e3",
   "execution_count": 8
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Witek\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\optim\\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100 | Train Loss: 0.3004033607808319 | Val Loss: 0.27279983903910665\n",
      "Epoch 2/100 | Train Loss: 0.26719636208302266 | Val Loss: 0.2635624563774547\n",
      "Epoch 3/100 | Train Loss: 0.26297915082525564 | Val Loss: 0.2632145821242719\n",
      "Epoch 4/100 | Train Loss: 0.26168077056472366 | Val Loss: 0.2608839776870367\n",
      "Epoch 5/100 | Train Loss: 0.26068737881409154 | Val Loss: 0.25883494317531586\n",
      "Epoch 6/100 | Train Loss: 0.259621803824966 | Val Loss: 0.25939259335801407\n",
      "Epoch 7/100 | Train Loss: 0.25818657794514216 | Val Loss: 0.2591487499507698\n",
      "Epoch 8/100 | Train Loss: 0.2573926646161724 | Val Loss: 0.25823143446767655\n",
      "Epoch 9/100 | Train Loss: 0.2569717545364354 | Val Loss: 0.2598579302430153\n",
      "Epoch 10/100 | Train Loss: 0.25729873353565064 | Val Loss: 0.25683086848742254\n",
      "Epoch 11/100 | Train Loss: 0.2561647710767952 | Val Loss: 0.25767815556075124\n",
      "Epoch 12/100 | Train Loss: 0.2555568842469035 | Val Loss: 0.25763831025845296\n",
      "Epoch 13/100 | Train Loss: 0.2567410088471464 | Val Loss: 0.25622209846167954\n",
      "Epoch 14/100 | Train Loss: 0.2555974733990592 | Val Loss: 0.2563420697241216\n",
      "Epoch 15/100 | Train Loss: 0.25527436145254084 | Val Loss: 0.2572782935725676\n",
      "Epoch 16/100 | Train Loss: 0.25591661015877853 | Val Loss: 0.2615093904975298\n",
      "Epoch 17/100 | Train Loss: 0.2572338722042135 | Val Loss: 0.2572701414291923\n",
      "Epoch 18/100 | Train Loss: 0.25535370248395045 | Val Loss: 0.2603039465643264\n",
      "Epoch 19/100 | Train Loss: 0.2551426587475313 | Val Loss: 0.2563601095934172\n",
      "Epoch 20/100 | Train Loss: 0.2543534195503673 | Val Loss: 0.2550088930371645\n",
      "Epoch 21/100 | Train Loss: 0.25444929100371694 | Val Loss: 0.25489967034475225\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001B[1;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[1;31mKeyboardInterrupt\u001B[0m                         Traceback (most recent call last)",
      "Cell \u001B[1;32mIn[9], line 40\u001B[0m\n\u001B[0;32m     38\u001B[0m \u001B[38;5;28;01mfor\u001B[39;00m x, y \u001B[38;5;129;01min\u001B[39;00m val_dataloader:\n\u001B[0;32m     39\u001B[0m     x, y \u001B[38;5;241m=\u001B[39m x\u001B[38;5;241m.\u001B[39mto(device), y\u001B[38;5;241m.\u001B[39mto(device)\n\u001B[1;32m---> 40\u001B[0m     preds \u001B[38;5;241m=\u001B[39m \u001B[43mmodel\u001B[49m\u001B[43m(\u001B[49m\u001B[43mx\u001B[49m\u001B[43m)\u001B[49m\n\u001B[0;32m     41\u001B[0m     loss \u001B[38;5;241m=\u001B[39m criterion(preds, y)\n\u001B[0;32m     42\u001B[0m     val_loss \u001B[38;5;241m+\u001B[39m\u001B[38;5;241m=\u001B[39m loss\u001B[38;5;241m.\u001B[39mitem()\n",
      "File \u001B[1;32m~\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\nn\\modules\\module.py:1736\u001B[0m, in \u001B[0;36mModule._wrapped_call_impl\u001B[1;34m(self, *args, **kwargs)\u001B[0m\n\u001B[0;32m   1734\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_compiled_call_impl(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)  \u001B[38;5;66;03m# type: ignore[misc]\u001B[39;00m\n\u001B[0;32m   1735\u001B[0m \u001B[38;5;28;01melse\u001B[39;00m:\n\u001B[1;32m-> 1736\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_call_impl(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)\n",
      "File \u001B[1;32m~\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\nn\\modules\\module.py:1747\u001B[0m, in \u001B[0;36mModule._call_impl\u001B[1;34m(self, *args, **kwargs)\u001B[0m\n\u001B[0;32m   1742\u001B[0m \u001B[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001B[39;00m\n\u001B[0;32m   1743\u001B[0m \u001B[38;5;66;03m# this function, and just call forward.\u001B[39;00m\n\u001B[0;32m   1744\u001B[0m \u001B[38;5;28;01mif\u001B[39;00m \u001B[38;5;129;01mnot\u001B[39;00m (\u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_backward_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_backward_pre_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_forward_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_forward_pre_hooks\n\u001B[0;32m   1745\u001B[0m         \u001B[38;5;129;01mor\u001B[39;00m _global_backward_pre_hooks \u001B[38;5;129;01mor\u001B[39;00m _global_backward_hooks\n\u001B[0;32m   1746\u001B[0m         \u001B[38;5;129;01mor\u001B[39;00m _global_forward_hooks \u001B[38;5;129;01mor\u001B[39;00m _global_forward_pre_hooks):\n\u001B[1;32m-> 1747\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m forward_call(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)\n\u001B[0;32m   1749\u001B[0m result \u001B[38;5;241m=\u001B[39m \u001B[38;5;28;01mNone\u001B[39;00m\n\u001B[0;32m   1750\u001B[0m called_always_called_hooks \u001B[38;5;241m=\u001B[39m \u001B[38;5;28mset\u001B[39m()\n",
      "Cell \u001B[1;32mIn[2], line 32\u001B[0m, in \u001B[0;36mMLP.forward\u001B[1;34m(self, x)\u001B[0m\n\u001B[0;32m     30\u001B[0m x \u001B[38;5;241m=\u001B[39m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39minit_layer(x)\n\u001B[0;32m     31\u001B[0m \u001B[38;5;28;01mfor\u001B[39;00m block \u001B[38;5;129;01min\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39mblocks:\n\u001B[1;32m---> 32\u001B[0m     x \u001B[38;5;241m=\u001B[39m \u001B[43mblock\u001B[49m\u001B[43m(\u001B[49m\u001B[43mx\u001B[49m\u001B[43m)\u001B[49m\n\u001B[0;32m     34\u001B[0m x \u001B[38;5;241m=\u001B[39m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39moutput_layer(x)\n\u001B[0;32m     35\u001B[0m \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39msoftplus(x)\u001B[38;5;241m.\u001B[39mflatten()\n",
      "File \u001B[1;32m~\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\nn\\modules\\module.py:1736\u001B[0m, in \u001B[0;36mModule._wrapped_call_impl\u001B[1;34m(self, *args, **kwargs)\u001B[0m\n\u001B[0;32m   1734\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_compiled_call_impl(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)  \u001B[38;5;66;03m# type: ignore[misc]\u001B[39;00m\n\u001B[0;32m   1735\u001B[0m \u001B[38;5;28;01melse\u001B[39;00m:\n\u001B[1;32m-> 1736\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_call_impl(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)\n",
      "File \u001B[1;32m~\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\nn\\modules\\module.py:1747\u001B[0m, in \u001B[0;36mModule._call_impl\u001B[1;34m(self, *args, **kwargs)\u001B[0m\n\u001B[0;32m   1742\u001B[0m \u001B[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001B[39;00m\n\u001B[0;32m   1743\u001B[0m \u001B[38;5;66;03m# this function, and just call forward.\u001B[39;00m\n\u001B[0;32m   1744\u001B[0m \u001B[38;5;28;01mif\u001B[39;00m \u001B[38;5;129;01mnot\u001B[39;00m (\u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_backward_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_backward_pre_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_forward_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_forward_pre_hooks\n\u001B[0;32m   1745\u001B[0m         \u001B[38;5;129;01mor\u001B[39;00m _global_backward_pre_hooks \u001B[38;5;129;01mor\u001B[39;00m _global_backward_hooks\n\u001B[0;32m   1746\u001B[0m         \u001B[38;5;129;01mor\u001B[39;00m _global_forward_hooks \u001B[38;5;129;01mor\u001B[39;00m _global_forward_pre_hooks):\n\u001B[1;32m-> 1747\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m forward_call(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)\n\u001B[0;32m   1749\u001B[0m result \u001B[38;5;241m=\u001B[39m \u001B[38;5;28;01mNone\u001B[39;00m\n\u001B[0;32m   1750\u001B[0m called_always_called_hooks \u001B[38;5;241m=\u001B[39m \u001B[38;5;28mset\u001B[39m()\n",
      "Cell \u001B[1;32mIn[2], line 11\u001B[0m, in \u001B[0;36mResidualBlock.forward\u001B[1;34m(self, x)\u001B[0m\n\u001B[0;32m      9\u001B[0m \u001B[38;5;28;01mdef\u001B[39;00m \u001B[38;5;21mforward\u001B[39m(\u001B[38;5;28mself\u001B[39m, x):\n\u001B[0;32m     10\u001B[0m     residual \u001B[38;5;241m=\u001B[39m x\n\u001B[1;32m---> 11\u001B[0m     out \u001B[38;5;241m=\u001B[39m \u001B[38;5;28;43mself\u001B[39;49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mln\u001B[49m\u001B[43m(\u001B[49m\u001B[43mx\u001B[49m\u001B[43m)\u001B[49m\n\u001B[0;32m     12\u001B[0m     out \u001B[38;5;241m=\u001B[39m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39mfc(out)\n\u001B[0;32m     13\u001B[0m     out \u001B[38;5;241m=\u001B[39m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39mgelu(out)\n",
      "File \u001B[1;32m~\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\nn\\modules\\module.py:1736\u001B[0m, in \u001B[0;36mModule._wrapped_call_impl\u001B[1;34m(self, *args, **kwargs)\u001B[0m\n\u001B[0;32m   1734\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_compiled_call_impl(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)  \u001B[38;5;66;03m# type: ignore[misc]\u001B[39;00m\n\u001B[0;32m   1735\u001B[0m \u001B[38;5;28;01melse\u001B[39;00m:\n\u001B[1;32m-> 1736\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_call_impl(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)\n",
      "File \u001B[1;32m~\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\nn\\modules\\module.py:1747\u001B[0m, in \u001B[0;36mModule._call_impl\u001B[1;34m(self, *args, **kwargs)\u001B[0m\n\u001B[0;32m   1742\u001B[0m \u001B[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001B[39;00m\n\u001B[0;32m   1743\u001B[0m \u001B[38;5;66;03m# this function, and just call forward.\u001B[39;00m\n\u001B[0;32m   1744\u001B[0m \u001B[38;5;28;01mif\u001B[39;00m \u001B[38;5;129;01mnot\u001B[39;00m (\u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_backward_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_backward_pre_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_forward_hooks \u001B[38;5;129;01mor\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_forward_pre_hooks\n\u001B[0;32m   1745\u001B[0m         \u001B[38;5;129;01mor\u001B[39;00m _global_backward_pre_hooks \u001B[38;5;129;01mor\u001B[39;00m _global_backward_hooks\n\u001B[0;32m   1746\u001B[0m         \u001B[38;5;129;01mor\u001B[39;00m _global_forward_hooks \u001B[38;5;129;01mor\u001B[39;00m _global_forward_pre_hooks):\n\u001B[1;32m-> 1747\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m forward_call(\u001B[38;5;241m*\u001B[39margs, \u001B[38;5;241m*\u001B[39m\u001B[38;5;241m*\u001B[39mkwargs)\n\u001B[0;32m   1749\u001B[0m result \u001B[38;5;241m=\u001B[39m \u001B[38;5;28;01mNone\u001B[39;00m\n\u001B[0;32m   1750\u001B[0m called_always_called_hooks \u001B[38;5;241m=\u001B[39m \u001B[38;5;28mset\u001B[39m()\n",
      "File \u001B[1;32m~\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\nn\\modules\\normalization.py:217\u001B[0m, in \u001B[0;36mLayerNorm.forward\u001B[1;34m(self, input)\u001B[0m\n\u001B[0;32m    216\u001B[0m \u001B[38;5;28;01mdef\u001B[39;00m \u001B[38;5;21mforward\u001B[39m(\u001B[38;5;28mself\u001B[39m, \u001B[38;5;28minput\u001B[39m: Tensor) \u001B[38;5;241m-\u001B[39m\u001B[38;5;241m>\u001B[39m Tensor:\n\u001B[1;32m--> 217\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[43mF\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mlayer_norm\u001B[49m\u001B[43m(\u001B[49m\n\u001B[0;32m    218\u001B[0m \u001B[43m        \u001B[49m\u001B[38;5;28;43minput\u001B[39;49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[38;5;28;43mself\u001B[39;49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mnormalized_shape\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[38;5;28;43mself\u001B[39;49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mweight\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[38;5;28;43mself\u001B[39;49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mbias\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[38;5;28;43mself\u001B[39;49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43meps\u001B[49m\n\u001B[0;32m    219\u001B[0m \u001B[43m    \u001B[49m\u001B[43m)\u001B[49m\n",
      "File \u001B[1;32m~\\anaconda3\\envs\\mpsi\\lib\\site-packages\\torch\\nn\\functional.py:2900\u001B[0m, in \u001B[0;36mlayer_norm\u001B[1;34m(input, normalized_shape, weight, bias, eps)\u001B[0m\n\u001B[0;32m   2890\u001B[0m \u001B[38;5;28;01mif\u001B[39;00m has_torch_function_variadic(\u001B[38;5;28minput\u001B[39m, weight, bias):\n\u001B[0;32m   2891\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m handle_torch_function(\n\u001B[0;32m   2892\u001B[0m         layer_norm,\n\u001B[0;32m   2893\u001B[0m         (\u001B[38;5;28minput\u001B[39m, weight, bias),\n\u001B[1;32m   (...)\u001B[0m\n\u001B[0;32m   2898\u001B[0m         eps\u001B[38;5;241m=\u001B[39meps,\n\u001B[0;32m   2899\u001B[0m     )\n\u001B[1;32m-> 2900\u001B[0m \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[43mtorch\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mlayer_norm\u001B[49m\u001B[43m(\u001B[49m\n\u001B[0;32m   2901\u001B[0m \u001B[43m    \u001B[49m\u001B[38;5;28;43minput\u001B[39;49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mnormalized_shape\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mweight\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mbias\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43meps\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mtorch\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mbackends\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mcudnn\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43menabled\u001B[49m\n\u001B[0;32m   2902\u001B[0m \u001B[43m\u001B[49m\u001B[43m)\u001B[49m\n",
      "\u001B[1;31mKeyboardInterrupt\u001B[0m: "
     ]
    }
   ],
   "source": [
    "from torch.optim import AdamW\n",
    "import torch.optim as optim\n",
    "\n",
    "lr = 1e-3\n",
    "weight_decay = 1e-5\n",
    "optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)\n",
    "\n",
    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(\n",
    "    optimizer, \n",
    "    mode='min',       \n",
    "    factor=0.5,\n",
    "    patience=5,   \n",
    "    threshold=1e-4,  \n",
    "    verbose=True \n",
    ")\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "num_epochs = 100\n",
    "criterion = MLP.criterion\n",
    "model = model.to(device)\n",
    "\n",
    "for epoch in range(num_epochs):\n",
    "    model.train()\n",
    "    train_loss = 0\n",
    "    for x, y in dataloader:\n",
    "        optimizer.zero_grad()\n",
    "        x, y = x.to(device), y.to(device)\n",
    "        y_pred = model(x)\n",
    "        loss = criterion(y_pred, y)\n",
    "        train_loss += loss.item()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "    \n",
    "    avg_train_loss = train_loss / len(dataloader)\n",
    "    \n",
    "    model.eval()\n",
    "    val_loss = 0\n",
    "    with torch.no_grad():\n",
    "        for x, y in val_dataloader:\n",
    "            x, y = x.to(device), y.to(device)\n",
    "            preds = model(x)\n",
    "            loss = criterion(preds, y)\n",
    "            val_loss += loss.item()\n",
    "    \n",
    "    avg_val_loss = val_loss / len(val_dataloader)\n",
    "    \n",
    "    scheduler.step(avg_val_loss)\n",
    "    \n",
    "    print(f\"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss}\")\n",
    "    \n",
    "    current_lr = optimizer.param_groups[0]['lr']\n",
    "    if current_lr < 1e-7:\n",
    "        print(\"Early stopping: LR zbyt niski.\")\n",
    "        break"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:33:29.424925Z",
     "start_time": "2026-01-27T10:32:36.601959Z"
    }
   },
   "id": "486c63375b0e3635",
   "execution_count": 9
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor(0.2538, device='cuda:0')\n"
     ]
    }
   ],
   "source": [
    "with torch.no_grad():\n",
    "    X_test, y_test = X_test.to(device), y_test.to(device)\n",
    "    preds = model(X_test)\n",
    "    test_loss = MLP.criterion(preds, y_test)\n",
    "    print(test_loss)"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:33:32.300852Z",
     "start_time": "2026-01-27T10:33:32.121137Z"
    }
   },
   "id": "e6e452a4a4f839ca",
   "execution_count": 10
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[5.2557, 6.4482],\n",
      "        [1.2151, 1.1320],\n",
      "        [2.0045, 2.3137],\n",
      "        [1.8117, 2.0117],\n",
      "        [3.6536, 3.8451],\n",
      "        [1.8387, 2.2417],\n",
      "        [3.7859, 4.9348],\n",
      "        [2.6821, 2.1956],\n",
      "        [2.5802, 2.0312],\n",
      "        [2.4701, 2.9482]], device='cuda:0')\n"
     ]
    }
   ],
   "source": [
    "print(torch.stack([preds[:10], y_test[:10]], dim=-1))"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:33:33.320041Z",
     "start_time": "2026-01-27T10:33:33.311040Z"
    }
   },
   "id": "b50c2d21f336c5fd",
   "execution_count": 11
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "data": {
      "text/plain": "tensor([6.8210, 1.3042, 2.2839, 1.8756, 4.1706, 2.0808, 4.3551, 2.9594, 2.7977,\n        2.8511], device='cuda:0')"
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test[:10, 0]"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:33:34.196019Z",
     "start_time": "2026-01-27T10:33:34.182019Z"
    }
   },
   "id": "d3de5ce5f1047f8e",
   "execution_count": 12
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [
    "relative_errors_model = (torch.abs(preds - y_test) / y_test).to('cpu').detach().numpy().copy()\n",
    "relative_errors_baseline = (torch.abs(X_test[:, 0] - y_test) / y_test).to('cpu').detach().numpy().copy()"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:33:34.951043Z",
     "start_time": "2026-01-27T10:33:34.931042Z"
    }
   },
   "id": "362678b0c6aa208d",
   "execution_count": 13
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "data": {
      "text/plain": "Text(0.5, 1.0, 'Baseline relative errors (sqr_avg = 0.3421599864959717)')"
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": "<Figure size 1700x700 with 2 Axes>",
      "image/png": "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qtOaNWs0f/589e3bVw8//LB+/vlnLVmyRKmpqW7JzKp4DqKjo71+lli0aJFSUlI0YMAAt/byxsXb4ofvv/9eU6dO1e233y5Jaty4se655x498sgjevHFF93GHH5iAGPMsGHDzKBBg0xsbKz561//6nH7W2+9Za666irTrVs3M3Xq1HM+32effWYiIyPNZ599Vun7DBkyxEyePPmcY6mMYcOGmWHDhlm6z9SpU03Pnj19FFHgGz16tHnsscf8HUatl5+fbzp37myefPJJV1txcbG5/fbbzT333FPm/d555x0TGRlp9u7d69Y+cuRIM2DAANfvTz/9tLn22mvLPE5eXp654oorzOOPP+7WPmzYMHP77beXeb+UlBQTHR1tVq9eXWafX375xfTs2dMkJiZajru0NWvWmLZt25rk5GRX2/Tp002vXr1MXl6eq+2rr74y11xzjfniiy+MMcYUFBSYdu3amVdffbXMYz/zzDOmffv25vjx46425/vbhx9+aIwx5q9//auJjIw0X375pdt9x48fb4YMGeL6fciQIeYPf/iDW58HH3zQ7b3mueeeMx06dDA//fSTq2379u1ux3/hhRdMdHS0+fHHH119Tp8+bQYMGOD2fE6ePNnjfKWd7TXglJiYaObOnXvWfgBQE5Q19/v1119NbGysx79r1eE///mPiYyMNG+88Ua1n/tcMI+ufsyjq0Zl59EbNmwwkZGR5ptvvnG1Of9+X375ZY/+27dvN3FxcaZTp05un70dDofp3LmzmT17tlv/zZs3m8jISPPVV1+52r744gtz4403mvj4eBMZGWmWLFnicZ5u3bqZhQsXlhl3ReajP/30k7n88svNjBkz3Pq8//77pm3btiYtLc0YY8z69etNZGSkOXLkiKvPhx9+aCIjI83nn3/uek7atm1r1q1b53asiRMnmgceeMAjvq+//trExsaaa665xu095fTp0yY+Pt7ce++9bv1TU1NNZGSk+eijj6r0OfDmX//6l4mMjDRvv/22W3tFxqWk06dPm1tvvdX8/ve/N4WFha72wsJCEx8fb955551y74/qQSkAuJx33nnq3r272+WqTlu2bFHfvn0VHOy+yLmwsFDPPvus+vXrp5iYGPXp00crV65UcXGxW7/169erb9++io2N1bBhw5SRkeFxjoyMDE2aNEnx8fHq0KGD7r77bh04cKDMeNu0aeNaDl9cXKyFCxeqV69eat++vXr16qW//OUvOnXqVJn3f+aZZ3T99ddr6dKlrssRTp48KenMt+IDBgxQ+/bt1aNHDz3zzDNyOBxlHuv48eOaM2eOevbsqfbt27suEXZeijVt2jRt3LhRP/zwg+uypZKXMP3444+6/PLLPS6ZOH78uKKjo/W3v/3N9ThXrlyp66+/Xu3bt1ffvn310ksvlRmXU2FhoZ588kl1795d7du314033uhxGW6vXr00f/583X333YqNjdWMGTNclzWsX79ePXv2VMeOHfXxxx9Lkj7++GPdcccd6tSpk7p06aLJkyfrv//9r+t4GzZsULt27fTaa6/pmmuuUXx8vNLS0vT9999rzJgx6tKlizp06KAhQ4ac9fKZb775Rtu3b/e4nPxsY37ixAlNnTpV8fHxio+P1/z587VkyRK3y8SHDx+uhx56SBMmTNAVV1yhe++996zPp1NBQYH+8pe/qE+fPmrfvr06duyoe++917Vib9OmTYqKitI333zjdr9//etfioqKcr2+Dx8+rPvuu08dO3bU1VdfrYULF2r69OnlXiY3bdq0ci8TKavMxt69e5WTk6Prr7/e1Waz2XT99dfr888/V0FBgdf7hYWFaciQIYqJiXFrb9mypb7//nvX7wcPHtTll19eZtz169fXyy+/7FpN6VSvXj0VFhZ6vY8xRo899phatWqle+65p8xjL1u2TMePH9ef/vQny3GXlJ2drUWLFukPf/iDOnTo4Irh3Xff1a233uq2CjMmJkYfffSRa4XCN998o9OnT5f7HNxxxx36+9//rsaNG7s9fkmu5+Dw4cNq1KiROnXq5HbfLl26aM+ePa73qsLCQoWFhbn1ueCCC3TixAnX7x999JE6deqkJk2auNoSEhLUsGFDffjhh5Kk5s2b65577lHz5s1dfex2u377299aGt+K9pGkG2+8Ua+//rqOHz9+1r4AUFOFhoYqJCTEbcWQw+HQypUrNXDgQMXGxuqKK67Q0KFD9dlnn7n6FBQUaPbs2br22mvVvn179evXz+NqgBMnTuhPf/qTrr76asXExOj222/3KNFSUulSANOmTdM999yjN954Q3379lX79u118803u977nazOwSXm0cyjz2AeXbF5dO/evfXyyy+7rWovPfdzOnnypGbOnKmHH37Y7Yog6Ux5gJtvvtltLKUz81pJbpeGjx07VhdffHGZ5TKOHz+uY8eOlTtnq8h89LvvvpPD4VDPnj09+hQXF+vf//632+MsOW91rtR1zlv/9a9/KSQkRLfddpvbsRYtWuRxyXxRUZGmTJmi4cOH63e/+53bbdnZ2Tpx4oR69Ojh1h4ZGanGjRtr+/btVfoclFZQUKA///nP6tGjh/r16+d229nGpbT169dr//79mjNnjurXr+9qr1+/vvr27asVK1ZU6DjwLUoBwM0NN9ygP/7xj27lAHJzc/Xhhx/qr3/9q9tEzBijMWPGKDk5WQ888IDatm2rzz//XIsWLdJ//vMfzZ07V5K0du1azZ07V3fffbeuvfZaffrpp3r00Ufdznv8+HENHTpUoaGhevTRRxUaGqo1a9bozjvv1Ouvv65WrVp5xOo8viStWrVKL7/8sqZOnapLLrlEe/fu1cKFC1WvXj1NmDChzMebkZGhHTt2aOHChTpx4oQaNWqkFStWaOHChRo2bJimT5+ur7/+Ws8884z++9//av78+R7HMMZo9OjROnnypB566CE1bdpUqampWrRokWbNmqXnn39e48aN0/Hjx3XgwAEtXbpUl156qfLy8lzHuOiiixQfH6/Nmzdr2LBhrvatW7fKGOO6fGD27NnasGGDRo8erbi4OH3xxReaP3++cnJydP/993t9jMYY3X///dq9e7cmTJigVq1aadu2bR6XcEtnLrW+9957dd9996lhw4auy6yXLl2qmTNnqqCgQHFxcXrzzTc1depUDRw4UKNHj3ZdWjFkyBBt3LjRdWmvw+HQCy+8oHnz5unnn3/W7373O9el3k8++aSCg4P14osvauzYsXr77bf129/+1utj2LRpkyIiInTFFVdUeMyLi4s1atQo/fDDD3r44YfVuHFjrVy5Ut99951HIurtt9/WTTfdpOXLl3t8KVCeKVOm6Msvv9SkSZN06aWX6siRI1q8eLEmT56szZs3q3fv3jrvvPO0efNmRUZGuu731ltvqU2bNmrXrp2OHz+uYcOG6cILL9SCBQvkcDi0ePFiZWRkuD3e0saNG1duTeTWrVt7bXdeRue8LN3pt7/9rRwOh77//nu3WJ2uvvpqXX311W5tp06d0o4dO9zO9fXXX+u3v/2thg4dqv379ys8PFy33HKLJk6cqHr16slut6tt27aSzrw2f/rpJ23YsEGffPKJHnvsMa8xb9myRXv37tWLL74ou93utU9GRoZefPFFjR49Wr/5zW8sx13SkiVLFBQUpD/+8Y+utqNHj+qXX37RxRdfrDlz5mjz5s3Kz89XQkKCZs2a5Xq/PHjwoKQzHyrHjBmjEydOKDY2VlOnTnUlaZs0aeJKchYWFurrr7/WY489pksvvVQJCQmSzlzi8+uvv+rkyZNq1KiRKw5nkvPo0aNq1KiR7rrrLs2YMUP/+Mc/1KtXLyUnJ2vjxo1uf9eHDx/WDTfc4PYY7Xa7WrRooW+//VbSmff+0n1OnjypL774QldddZUr1m+//VYtWrTQzTffrMOHDysiIkLDhg3TiBEjXEmFs70GnHr16iWHw6Ft27Z5vbQLAGoaY4yrPp8xRidOnNCaNWtUVFTkKlMjSU899ZRefvllTZ48WVFRUTp27JieffZZTZw4Udu3b1doaKjmz5+vjz76SFOnTlXTpk314Ycf6sknn9QFF1ygW2+9VYWFhbr77ruVnZ2tBx98UM2aNdMbb7yhUaNGafXq1eratWuFYt63b58yMzM1YcIEhYWFafHixRo/frw+/PBDNWrUqFJzcCfm0cyjJebR0tnn0eeff746duwo6UxCMD09XU888YQaN26s/v37u/WdO3euWrVqpaFDh2rVqlVut4WHh2vmzJkex//Xv/7lEffatWtdi5C8cc5Zt2/frscff1yZmZlq06aNHnzwQVcZrIrMR50LBUovnCrZR5L69++vlStX6rHHHtMjjzzi+sIiIiLCNVd3ziG/+OILPfXUU/rmm2/UvHlzjR492mOu+Oyzz+r06dOaMGGCRo4c6fE8BQcHe8R08uRJ5eTkuBLQVfUclGyXpBdffFHHjh1zfalT0tnGpaRff/1VS5Ys0c033+wqu1BSv3799Morr+jbb7/1SC6jmvlnoSxqGuclOfn5+eaKK65wKwewYcMG0717d1NcXGx69uzpuhzBeSnpW2+95XasZ5991nWpQ3Fxsenatav54x//6NbnT3/6k9tl/U8//bSJiYkxR48edfUpLCw01113nRk/frwxpvzyASNGjPBY6v/SSy+ZN998s8zHvGTJEhMZGem6hNcYY3JyckxsbKz505/+5Nb31Vdfdbt8o+QlTD/++KMZPny423GMMWbu3Lmmffv2rt9LX8JU+vKtN954w0RFRZkffvjB1eeOO+4wI0eONMYYk56ebqKiosyKFSvczrNw4UITExPjdllxSR999JGJjIw0mzdvdmt/6KGHzDXXXGNOnTpljDGmZ8+epnfv3m59nM/5s88+62pzOBzmmmuuMSNGjHDre+TIERMdHW2eeOIJ1+OJjIx0G4PMzEwTGRlp/vnPf7racnJyzPz5890ujSnttttuM2PHjnVrO9uYv//++yYyMtJ88MEHrtt/+eUXEx8f7zYOw4YNMx06dHC7tKIiCgsLzYgRIzye1xdeeMFERkaazMxMY8yZcS/5vObm5prY2FjXOC5atMjExMS4XYJ99OhREx0dbfkyuYpYsWKFiYyMdI2708cff2wiIyPNrl27KnysefPmmcjISNclKD/99JOJjIw0V199tdm4caP5/PPPzaJFi0x0dLSZNGmSx/03bdpkIiMjTWRkpElMTDT5+flez3PLLbeYoUOHlhvL/PnzTVxcnDlx4oTluEvKzs42MTEx5umnn3Zr37t3r4mMjDTXXHONGTt2rPn3v/9t3nzzTXPttdea66+/3vz666/GGGPmzJljIiMjzcMPP2w+++wzs3XrVleZla+//trjfH369DGRkZEmNjbWVQbAGGMOHTpkoqOjzV133WW++eYbc/LkSfOPf/zDdO7c2e19q7Cw0EybNs31PEZGRpoRI0aYoqIi17Hat2/v8XiMMWbo0KEef0NODofDjB8/3lx++eWuMgpfffWViYyMNH369DFvv/22+eSTT8zcuXNNVFSU6/hWXwM333yzmThxotcYAKAmGTZsmNt7bcmf5557zq3vpEmTzN/+9je3Nmdpmj179hhjjOnbt6+ZOXOmW5+lS5e65i2vvPKKiYyMdCtJU1xcbO68804zaNAgY4z3uWRkZKT5z3/+Y4w5Mwcpfentzp07TWRkpNm6dasxpmJzcG+YRzOPNoZ5tJOVefSIESNMZGSkadu2rUcZj3fffddcccUVrr/Hkp+9y5KcnGxiYmLM6NGjy+zj7ZLz1atXm8jISDNy5Ejz0Ucfmffff9+MGDHCtG3b1jUnreh89A9/+IO58sorzbvvvmtycnLM/v37zaBBg0z79u3N9OnTXef817/+ZWJjY13vnVdeeaXb/HjUqFGmS5cu5qqrrjJr1641n3zyiZk5c6aJjIw069evd/Xbu3evad++vWuO6q28yOTJk010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vN7uHF9vq2ccDofrA+CWLVsUHh7uqpXqVFhY6ErWfvbZZyosLHSbNAYHB7tt0JCRkaGPP/5YN998s9uYHDhwQBdccIFatGihvXv3SpLHhz3nhP3QoUPq2bOnfve733ls1OVwOHT06FG3Vbs//vij7r33Xh09elRPP/20R82vAwcOKDk5WUOHDnX7oFVyfCvyGigpJyfH8gcBAKhp2rdvr9dee01Hjx5VSEiIRo0apaioKG3evFktW7ZUUFCQduzYoXfeecd1n/r162vs2LEaO3asMjIy9MEHH2jZsmWu+pLnn3++LrvsMj311FNez9miRYtyE0cVda5z8JKYR58d8+gz6to8+quvvtLRo0c96tlHR0drz549kuR6fyj9ufuHH37Qm2++6Zbsff7555WUlKT4+Hg9++yzFUrolrZjxw4VFBSob9++bu0l56wVmY9KZ65wbdWqldq2bev6W0tJSVFxcbFrjv/DDz94rApv0KCB2rdvr0OHDkk68/dkjNGpU6cUEhLi6udMoDZo0EDvvPOOfvjhB6+rkqOjo7VgwQINGjRIH3zwgcLDw9WpUyfXSvmffvpJP/74oyumqnwOpDNXfX300UdVslrVWY+79AZcpTGXrhnOvlYddU79+vXVu3dvvfPOO3r77bfL/GY3Pj5ep0+f1tatW93anZcNderUSZdddpn+7//+z6OP8zKPksdyFl2OiYlx/fzjH//Q66+/XuaGNU5Dhw7Vn//8Z0nShRdeqEGDBunOO+9UTk6OpW9wOnTooHr16unYsWNucQQHB+vpp5/2usx+z549Ki4u1vjx412TQYfDoU8++UTS/1YEVOTSkISEBDVt2lQvvvii9u/fr5tvvtl1mzNZ8vPPP7vFdvz4cS1evLjMiU58fLzy8vJkjHG73zfffOMq+m3F7373O0VEROitt95ya//Pf/6j5OTkMr/Rl848V1dffbW++uor2Ww2XX755XrwwQcVGRnpUVy8pN/85jduO6VKZx/zq6++WqdPn3YVc5fO/GNXcrJ2Lvbt26fCwkIlJibq0ksvdX2D7Dx+ycuvb775Zv3444969tlnZbfb3ZJZV155pZKTk5WVleVqy8zMVHJycrnndyYSy/opvbGAU1xcnM477zy3D3fGGG3bts11ybY3p0+f1pgxY5SSkqKFCxd6TT5mZWVp1qxZHn/vW7ZsUVhYmKKjo1VQUKCpU6e6vuV1cu6SW7KY+4kTJ/Tdd9+V+5qSVO7rriJxO5WVqJTOrMa48sor9e6777olVz/99FPl5eW5/j7Xr1+vWbNmufU5duyYdu/e7ZoM/+Mf/9CUKVPc3ptyc3O1Z88e1+P/6aefNHPmTLfL7bKysrR582b16tVLNpvNtTL2yy+/dIt19+7dkuSa/F1zzTX64osvdPz4cVefjz76SHl5ea7Vpbm5ubr77ruVmZmpv/71rx5JVenMrsJz5szx2JF6y5Yt+s1vfqMWLVpU6DVQ0o8//ljhy90AoKb66quvZLfbdckllyg9PV0nTpzQXXfdpdatW7vmf84NYIuLi10f5F944QVJZ5J6d955pwYMGOCaD8XHx+u///2vLrzwQrd/3z/++GOtXr36rHPjijrXOXhJzKPPjnn0GXVtHv3hhx/q4YcfdhsHh8Ohzz77zDX3e/311z1+IiIi1LNnT73++uuuOdT69ev15JNPqn///lq9enWlkqrSmc3dpk+f7vbaz8vL0/bt211z1orMRyVp+fLlWrlypdvx//a3v+n88893Hatly5bavXu329gWFhZq//79rjmrc9HB5s2b3Y71/vvvKyoqSmFhYVq+fLnH8xQdHa3o6Gi9/vrrrhKEzueppDVr1shut7v6VOVzIJ2ZK+fn56tTp07lPvcVkZycrHbt2pW7ev/kyZPKz88v84shVB9WrMKrG264QaNHj1ZQUJDXnQelMytbu3TpopkzZ+rYsWNq27atdu7cqVWrVumWW25x7Ur40EMPafLkyZo5c6b69eun5ORkvfzyy27Huueee/SPf/xD99xzj0aMGKHGjRtry5YtevXVV89aN0g684/qCy+8oKZNmyouLk7Hjh3TX//6V8XHx3uskCpP48aNNWrUKC1evFi5ubnq0qWLjh07psWLF8tms3ld0u/8tvKxxx7TrbfeqpMnT2rdunWuy37z8vIUFham8PBwZWdna8eOHWXWYLLb7RowYIDWrl2r5s2bu12CEhUVpZtuukmPPvqofvjhB7Vv317ffvutFi5cqBYtWnhdGSCd+Qfqyiuv1Lhx41z1Db/66itX3Rgrz490ZmI7adIkTZ8+XZMnT9ZNN92kn3/+WUuXLlWjRo107733lnlf5z8OU6ZM0fjx49W0aVN98skn+vrrr3XXXXeVeb9rrrlGf//732WMcf3jdbYx79q1q7p3766ZM2cqOztbLVq00Nq1a12Xgpyr6OhoBQcHKykpSSNGjFBRUZE2bNig7du3S5LbbrWRkZG6/PLL9fe//139+/d3m6zdddddWrdunUaOHOmqrbZs2TKdOnXKa22ncxUaGqoRI0bo2WefVb169RQXF6c33nhD+/fvd1uN8eOPP7q+0a1fv77WrVunL7/8UkOGDNFFF13kMWG94oor1KlTJ3Xt2lWPP/64CgoK1Lp1a23fvl0vvfSSpk2bpvDwcIWHh+vWW2/Vs88+q+DgYLVr105ffvmlVq5cqdtuu81tN1Pn6s2ydmaVzkxK09PTXTWPSqtI3CXPV79+fbdadCVNmjRJw4cP13333acRI0bop59+0lNPPaUOHTq46seNGzdO9957r8aNG6e77rpLJ0+e1NKlS3XBBRdoxIgRks5cfrd161aNHTtWI0eOVFFRkVatWqVff/1V48ePl3RmtULHjh01e/ZsTZkyRXa7XYsWLZLdbnf1adeunfr27avHH39cJ0+eVIcOHZSWlqZnnnlG0dHRuv766yVJd9xxh9auXat7771XDzzwgE6cOKGkpCRde+21rg9wS5Ys0Xfffafx48crODjY7XmqX7++61yrV6/W1KlT9cc//lHNmjXTW2+9pffff19LlixRUFBQhV4DTr/88osOHTrkel4AoKbLzc11e38sKirS+++/rzfeeENDhgxRkyZNVK9ePYWFhem5555TcHCwgoOD9c4777i+UMzPz1eDBg0UHR2tpUuXql69eoqKitK3336rjRs3ulZODRo0yPXePWbMGP3f//2fPvnkE61atUrDhg0r90oOK851Dl4S8+izYx59Rl2bRw8dOlTr16/X6NGj9cADD6hevXr6+9//rm+++cZVa9bblVf169fXBRdc4LotKytLCxYs0G9+8xvdeeedHqUjLr300gq/Jp3z0fvuu0+jR49WcXGxVq1apfz8fEvzUUkaPny4Zs2apTZt2iguLk5btmzRW2+9pdmzZ7sSvxMnTtT999+viRMn6rbbblNRUZHWrFmjY8eOuepSd+nSRT179tSCBQuUn5+vNm3a6M0339Tu3bu1bNkySe6LMJycq9BLPofOVfDz589Xr1699Omnn2rFihW67777XHP9qnwOpP99dilZ9qqyvvnmG48r4Epzruo/Wz9Ug2rbJgs1Wumd9IqKisyVV15pbrrpJrd+pXcmzMvLM48//rjp1q2biY6ONn379jWrV682DofD7X6bN282AwYMMO3btzeDBg0yb731lomMjDSfffaZq8+RI0fMhAkTzJVXXmliY2PNTTfdZF577TXX7c6dNUvex+nUqVNmyZIlpnfv3qZ9+/ama9euZsaMGWXu8GnM/3Yz9Wbt2rXmhhtuMNHR0ebqq682kydPdttltPTztXbtWnPdddeZ9u3bmx49epipU6eabdu2mcjISLN9+3ZjjDGpqammX79+Jjo62qxYscJjN1Onffv2mcjISNeuoKUf59KlS811111noqOjzbXXXmtmzZplfv755zIfpzHG/Prrr2b+/Pnm2muvNdHR0aZXr17mL3/5iykoKHD18bbrZHnP+datW80tt9xioqOjTZcuXcxDDz1kMjIyXLeX3pnW6dtvvzUPPPCA6dq1q4mOjjYDBgxw2+XRm6+//tpjZ9yKjHl+fr7585//bLp06WKuuOIKM2PGDDNx4kSP3Uwru2vo22+/bQYMGGBiYmJMQkKCeeCBB8zOnTtNVFSUWbt2rVtf5y6nztdDSd988425++67TWxsrOnatat55plnzJAhQ8rd3fNcOBwO8+yzz5ru3bubmJgYc8stt3jE5fz7cI7fHXfcUeZuyCX/jn755RezYMEC07NnT9O+fXtzww03mFdffdXt2IWFhWbZsmWmT58+Jjo62vTu3dusXLnS6/tGZGSkSUtLK/OxZGdnm8jISPP3v//d6+0VjdsYY2bNmmWuvvrqcp+7Xbt2mWHDhpnY2FgTHx9vHnnkEXPy5Em3Pp988on5wx/+YDp27Gg6d+5sHnzwQbf3D2OM2b9/vxkxYoS58sorTVxcnBk9erRJTU1165OVlWUmTZpk4uPjTXx8vBk/frzHrqCFhYVm0aJFpmfPniY6Otpcf/315oknnjC5ublu/VJTU91eY48++qj55ZdfXLd37969zOeo5N9LZmammT59uunWrZtp3769ueWWW8y2bdvczlWR14AxZ8Y3JibmrO9fAFATDBs2zOP9MSYmxgwYMMAsX77cFBUVufp+9tlnZtCgQa733BEjRpgvv/zSxMXFueZ3v/zyi5k7d67p0aOHa073+OOPm/z8fNdxsrOzzfTp003Xrl1N+/btTd++fc2qVatc/16WnkuWnntN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"
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def square_mean(data):\n",
    "    return np.sqrt(np.square(data).mean())\n",
    "\n",
    "fig, ax = plt.subplots(1, 2, figsize=(17,7))\n",
    "ax1, ax2 = ax\n",
    "quant1 = np.quantile(relative_errors_baseline, 0.02)\n",
    "quant2 = np.quantile(relative_errors_baseline, 0.98)\n",
    "ax1.hist(relative_errors_model, bins=100, density=True, range=(quant1, quant2))\n",
    "ax1.set_title(f\"Model's relative errors (sqr_avg = {square_mean(relative_errors_model)})\")\n",
    "\n",
    "ax2.hist(relative_errors_baseline, bins=100, density=True, range=(quant1, quant2))\n",
    "ax2.set_title(f\"Baseline relative errors (sqr_avg = {square_mean(relative_errors_baseline)})\")"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:35:46.809634Z",
     "start_time": "2026-01-27T10:35:46.425028Z"
    }
   },
   "id": "4f2da1ba74dd4b2a",
   "execution_count": 22
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.2546657919883728\n"
     ]
    }
   ],
   "source": [
    "with torch.no_grad():\n",
    "    X, y = X.to(device), y.to(device)\n",
    "    total_preds = model(X)\n",
    "print(torch.sqrt(torch.mean(torch.square((total_preds - y) / y))).item())"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:34:07.848808Z",
     "start_time": "2026-01-27T10:34:07.451732Z"
    }
   },
   "id": "121864dabf4197ce",
   "execution_count": 20
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3413205146789551\n"
     ]
    }
   ],
   "source": [
    "baseline_preds = X[:, 0]\n",
    "print(torch.sqrt(torch.mean(torch.square((baseline_preds - y) / y))).item())"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:34:07.863808Z",
     "start_time": "2026-01-27T10:34:07.849809Z"
    }
   },
   "id": "fe4f64c5f0d3c0b3",
   "execution_count": 21
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [
    "model = model.to('cpu')\n",
    "model.eval()\n",
    "torch.save(model.state_dict(), f\"../residual_mlp.pt\")"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T10:38:27.931536Z",
     "start_time": "2026-01-27T10:38:27.910021Z"
    }
   },
   "id": "cc1e53912c0af644",
   "execution_count": 23
  },
  {
   "cell_type": "code",
   "outputs": [
    {
     "data": {
      "text/plain": "<Figure size 640x480 with 6 Axes>",
      "image/png": "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x+7XKpw+fRolJSVQKpXStOjoaKSlpUGj0egt++yzz+LNN98st447d+5YPE5zMqXNAHDr1i3MmTMH06ZNs2aYZmVKmw8dOoQnnngCcrlcmrZhwwbExcVZLd7qMqW9/v7+yMjIQEpKCjQaDb799lv4+vri4YcftnbYZAHMcaWcOce5Wn4DXDfHuds6AHuXk5OD2rVr6z1su169eigqKkJeXh7q1KkjTW/atKnee9PT07F//37069fPavGagyltBoAPPvgAzz77LJo3b27tUM3GlDZnZWUhIiIC77zzDn7++Wc0atQIEyZMQHR0tC1Cvy+mtLdbt274+eef8fzzz0Mul8PNzQ1LlixBrVq1bBE6mRlzXClnznGult8A181xPFNXBZVKpfehACC9Li4urvB9N2/exOjRo9GqVSs88cQTFo3R3Exp8++//46UlBSMHDnSavFZgiltLigoQHJyMgICAvD555/j0UcfxZAhQ3Dt2jWrxVtdprT31q1byMnJwbvvvot169ahV69emDRpEv766y+rxUuWwxxXyplznKvlN8B1cxyLuip4eXmV+wBoX3t7ext8T25uLgYNGgQhBBYsWOBwD/M2ts2FhYV49913MWXKlAr3haMw5fcsl8sRGhqKxMREhIWFYdy4cXjkkUewefNmq8VbXaa096OPPkJwcDBeeOEFtGzZEtOnT4ePjw82bNhgtXjJcpjjSjlzjnO1/Aa4bo5zrL9EGwgMDMStW7dQUlIiTcvJyYG3tzf8/PzKLX/jxg288MILKC4uxldffVXuNL4jMLbNR48eRVZWFhITE6FUKqWxC6+88greffddq8ddHab8ngMCAhAUFKQ37ZFHHnGoI1lT2nvixAm0aNFCeu3m5oYWLVrg6tWrVouXLIc5rpQz5zhXy2+A6+Y4FnVVCA0Nhbu7O1JTU6VpKSkpUCgU5Y5OCwoKMHToULi5uWHlypUIDAy0crTmYWybIyIi8NNPP2HTpk3SDwDMmDEDr732mpWjrh5Tfs9RUVE4c+aM3rRz586hUaNG1gjVLExpb/369ZGZmak37fz582jcuLE1QiULY44r5cw5ztXyG+DCOc7Wl986gnfeeUd0795dpKWliR07dohWrVqJH3/8UQghRHZ2tlCpVEIIIT755BMREREh0tLSRHZ2tvRz+/ZtW4Z/X4xtc1mOeLm/lrFtvnz5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b2/06tULb7/9NgAgMDCw3Pvq1q2LEydO6E2rUaOG9P+8vDwAwHvvvYf33nuv3Puzs7MBADdv3sSUKVOwc+dOyGQyNGnSRCoghRAASnNx7dq1y60jICAAubm599Fqsncs6sjsatWqhZycnHLTtdNq166NwMBA3Lp1CxqNRq+w++uvv0zenqenJ1599VW8+uqruHr1Kn755Rd8+umnGDt2LL777juD7/n5558xd+5cDB8+HF27dtWb98ADD6BGjRr46quvDL63SZMmJsdIRM4nKCgIc+bMgVqtxtGjR7F582asXr0aDz/8MADg1q1b5d6Tm5tb6cGrn58fAGD8+PFo3bp1ufnaHoU333wT586dwxdffAGlUglPT0+oVCqsW7dOWrZ27dq4ePFiuXVoC0dyPrz6lczu0UcfxZEjR3DlyhW96Vu2bEFAQACaNGmC1q1bo6SkBD///LM0XwiBnTt3mrStwsJCPPXUU1i+fDkAoGHDhnjhhRfQvXt3XL161eB7zp49izfffBOxsbF4/fXXy81v3bo1CgoKIISAQqGQfs6ePYukpCS9LhEick3bt29H27ZtkZOTA7lcLvUO+Pn5SbnnwoULemOEb9y4gSNHjuDf//53hesNCgpC3bp1cfnyZb38ExgYiI8//hgnT54EAKSkpODJJ59EmzZt4OnpCQDYs2cPAECj0QAA2rZti8uXL0vdskDpGT7tFbrkfHimjszupZdewpYtWzB48GCMGjUK/v7+2LRpEw4cOID3338fbm5uePTRR9GuXTtMnjwZubm5aNiwIb755hucOXMGMpnM6G15e3sjPDwcixYtgoeHB0JCQnD+/Hls3LgRTz31VLnl8/LyMGLECNSoUQPDhw/H8ePHpQQIAA8//DDi4uLw6KOPYuTIkRg5ciSaNm2Ko0ePYsGCBWjfvj3q1Kljlv1ERI6rVatW0Gg0SEhIwLBhw1CzZk388MMPuHPnDp588kls2rQJQgiMGDECY8aMgVwux6JFi1CrVq1Kn+Ygl8sxZswYvPvuu5DL5ejYsSNu376NTz/9FDdu3EB4eDiA0osutm7divDwcDz44IM4fPgwkpOTIZPJoFKpAAC9evXCV199hVGjRmHMmDHw9fXFZ599ppfzyLnIhLbznagaQkJCMGrUKIwePRoAkJWVhY8//hi//fYb7t27hxYtWuCVV17BE088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"
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.plotting.hist_frame(df[(df['sigma'] <= quant2) & (df['target'] <= quant2)].loc[:, ['sigma', 'target', 'size', 'timefunc', 'log_size', 'spread']], bins=100)\n",
    "plt.tight_layout()"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2026-01-27T11:07:22.328959Z",
     "start_time": "2026-01-27T11:07:21.266687Z"
    }
   },
   "id": "49e3366b3456fdfc",
   "execution_count": 32
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [],
   "metadata": {
    "collapsed": false
   },
   "id": "918b57ad3cabb715"
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
