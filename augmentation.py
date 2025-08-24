from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    # brightness_range=[0.7, 1.3],
    # channel_shift_range=50.0  # <-- color thoda change karega
)

train_generator = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)
