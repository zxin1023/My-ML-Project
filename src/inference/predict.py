def predict(model, dataset):
    model.eval()
    with torch.no_grad():
        pred = model(dataset)
    return pred
