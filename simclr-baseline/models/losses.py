def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=torch.bool)).to(z.device)
    exp_sim = torch.exp(sim) * mask
    positives = torch.cat([torch.diag(sim,batch_size), torch.diag(sim,-batch_size)], dim=0)
    return (-torch.log(torch.exp(positives)/exp_sim.sum(dim=1))).mean()
