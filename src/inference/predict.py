



def predict(model, dataset):
    model.eval()
    with torch.no_grad():
        predictions = model(dataset)
    return predictions1