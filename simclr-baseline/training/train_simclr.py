from config import *
from data.metadata import process_commonvoice_metadata
from data.dataset import SimCLRCommonVoiceDataset, collate_fn
from models.encoder import SmallCNNEncoder
from models.projection import ProjectionHead
from models.losses import nt_xent_loss

# -------------------- DATALOADERS --------------------
train_dataset = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split='train', duration=DURATION, return_metadata=False)
val_dataset   = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split='val', duration=DURATION, return_metadata=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=lambda b: collate_fn(b, False))
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, False))

# -------------------- TRAINING --------------------
encoder = SmallCNNEncoder().to(DEVICE)
proj    = ProjectionHead().to(DEVICE)
optimizer = torch.optim.Adam(list(encoder.parameters())+list(proj.parameters()), lr=LR)

start_epoch = 0
if RESUME and LAST_CHECKPOINT.exists():
    ckpt = torch.load(LAST_CHECKPOINT, map_location=DEVICE)
    encoder.load_state_dict(ckpt['encoder'])   
    proj.load_state_dict(ckpt['proj'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']+1
    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    encoder.train(); proj.train(); running_loss=0
    for s1,s2 in tqdm(train_loader, desc=f"Train epoch {epoch}"):
        s1, s2 = s1.to(DEVICE), s2.to(DEVICE)
        z1, z2 = encoder(s1), encoder(s2)
        p1, p2 = F.normalize(proj(z1), dim=-1), F.normalize(proj(z2), dim=-1)
        loss = nt_xent_loss(p1,p2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch} avg loss: {running_loss/len(train_loader):.4f}")

    # validation
    encoder.eval(); proj.eval(); val_loss=0
    with torch.no_grad():
        for s1,s2 in val_loader:
            s1,s2 = s1.to(DEVICE), s2.to(DEVICE)
            z1,z2 = encoder(s1), encoder(s2)
            p1,p2 = F.normalize(proj(z1), dim=-1), F.normalize(proj(z2), dim=-1)
            val_loss += nt_xent_loss(p1,p2).item()
    print(f"Validation loss: {val_loss/len(val_loader):.4f}")

    ckpt = {'encoder':encoder.state_dict(), 'proj':proj.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(ckpt, OUTPUT_DIR/f"checkpoint_epoch_{epoch}.pt")
    torch.save(ckpt, LAST_CHECKPOINT)
