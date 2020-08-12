import torch


class LSTMAE(torch.nn.Module):
    def __init__(self, nin, nh, nl, nout, nlayers, do):
        super(LSTMAE, self).__init__()
        self.nh = nh
        self.nl = nl
        if nlayers >= 2:
            self.enc = torch.nn.LSTM(input_size=nin, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
            self.dec = torch.nn.LSTM(input_size=nl + 1, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
        else:
            self.enc = torch.nn.LSTM(input_size=nin, hidden_size=nh, num_layers=nlayers, batch_first=True)
            self.dec = torch.nn.LSTM(input_size=nl + 1, hidden_size=nh, num_layers=nlayers, batch_first=True)
        self.fcd = torch.nn.Linear(nh, nout)
        self.fce = torch.nn.Linear(nh, nl)
        self.do = torch.nn.Dropout(p=do)

    def forward(self, x, seq_len):        
        z = self.encode(x, seq_len)
        pred = self.decode(x[:, :, 0], z)  # index: 0-time, 1-flux, 2-flux_err
        return pred, z

    def encode(self, x, seq_len):
        n, _, _ = x.shape
        x, (_, _) = self.enc(x)
        x = x[torch.arange(n), (seq_len - 1).type(dtype=torch.long)]
        x = self.fce(x)
        return x

    def decode(self, dt, z):
        n, l = dt.shape
        z = self.do(z)
        x_lat = torch.zeros((n, l, self.nl + 1)).to(dt.device)
        new_z = z.view(-1, self.nl, 1).expand(-1, -1, l).transpose(1, 2)
        x_lat[:, :, :-1] = new_z
        x_lat[:, :, -1] = dt
        output, (_, _) = self.dec(x_lat)  # input shape (seq_len, batch, features)
        output = self.fcd(output).squeeze()
        return output.squeeze()


class GRUAE(torch.nn.Module):
    def __init__(self, nin, nh, nl, nout, nlayers, do):
        super(GRUAE, self).__init__()
        self.nh = nh
        self.nl = nl
        if nlayers >= 2:
            self.enc = torch.nn.GRU(input_size=nin, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
            self.dec = torch.nn.GRU(input_size=nl + 1, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
        else:
            self.enc = torch.nn.GRU(input_size=nin, hidden_size=nh, num_layers=nlayers, batch_first=True)
            self.dec = torch.nn.GRU(input_size=nl + 1, hidden_size=nh, num_layers=nlayers, batch_first=True)
        self.fcd = torch.nn.Linear(nh, nout)
        self.fce = torch.nn.Linear(nh, nl)
        self.do = torch.nn.Dropout(p=do)

    def forward(self, x, seq_len):        
        z = self.encode(x, seq_len)        
        pred = self.decode(x[:, :, 0], z)  # index: 0-time, 1-flux, 2-flux_err
        return pred, z

    def encode(self, x, seq_len):
        n, _, _ = x.shape
        x, _ = self.enc(x)
        x = x[torch.arange(n), (seq_len - 1).type(dtype=torch.long)]
        x = self.fce(x)
        return x

    def decode(self, dt, z):
        n, l = dt.shape
        z = self.do(z)
        x_lat = torch.zeros((n, l, self.nl + 1)).to(dt.device)
        new_z = z.view(-1, self.nl, 1).expand(-1, -1, l).transpose(1, 2)
        x_lat[:, :, :-1] = new_z
        x_lat[:, :, -1] = dt
        output, _ = self.dec(x_lat)  # input shape (seq_len, batch, features)
        output = self.fcd(output).squeeze()
        return output.squeeze()
        