from tracker import yolo_tracking

model = yolo_tracking
results = model.val(data='data.yaml')
print(results.box.map50)