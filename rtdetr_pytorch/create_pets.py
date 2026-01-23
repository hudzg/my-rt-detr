import os
import xml.etree.ElementTree as ET

def create_pets_filelists(root_dir, output_dir=None):
    if output_dir is None:
        output_dir = root_dir

    # Standard paths for Oxford-IIIT Pet Dataset
    img_dir = os.path.join(root_dir, "images")
    ann_dir = os.path.join(root_dir, "annotations", "xmls")
    
    # The dataset usually comes with list files in annotations/
    list_dir = os.path.join(root_dir, "annotations")

    # -------------------
    # 1. Helper to write files
    def write_list(txt_filename, source_txt_path, out_filename):
        print(f"Processing {txt_filename}...")
        
        # Read the source list provided by the dataset (trainval.txt or test.txt)
        # Format in source is usually: image_name class_id species breed_id
        ids = []
        if os.path.exists(source_txt_path):
            with open(source_txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        ids.append(parts[0]) # The first part is the filename without extension
        else:
            # Fallback if specific txt files don't exist, scan directory (naive split)
            print(f"Warning: {source_txt_path} not found. Scanning xmls directory.")
            all_xmls = [os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".xml")]
            # Just grabbing them all if source list missing is risky, but handles basic cases
            ids = all_xmls 

        final_path = os.path.join(output_dir, out_filename)
        valid_count = 0
        with open(final_path, "w") as f:
            for img_id in ids:
                # Oxford Pets are usually jpg
                img_path_rel = f"images/{img_id}.jpg"
                ann_path_rel = f"annotations/xmls/{img_id}.xml"
                
                # Check if xml exists (some images in this dataset might not have XMLs if downloaded partially)
                abs_ann_path = os.path.join(root_dir, "annotations", "xmls", f"{img_id}.xml")
                
                if os.path.exists(abs_ann_path):
                    f.write(f"{img_path_rel} {ann_path_rel}\n")
                    valid_count += 1
        
        print(f"✅ Created {final_path} ({valid_count} lines)")
        return ids

    # -------------------
    # 2. Create trainval.txt (used for training)
    trainval_source = os.path.join(list_dir, "trainval.txt")
    write_list("trainval.txt", trainval_source, "trainval.txt")

    # -------------------
    # 3. Create test.txt (used for validation/test)
    test_source = os.path.join(list_dir, "test.txt")
    write_list("test.txt", test_source, "test.txt")

    # -------------------
    # 4. Create label_list.txt
    # There are 37 categories in Oxford-IIIT Pet Dataset
    label_list = [
        "Abyssinian", "American_Bulldog", "American_Pit_Bull_Terrier", "Basset_Hound", 
        "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British_Shorthair", 
        "Chihuahua", "Egyptian_Mau", "English_Cocker_Spaniel", "English_Setter", 
        "German_Shorthaired", "Great_Pyrenees", "Havanese", "Japanese_Chin", 
        "Keeshond", "Leonberger", "Maine_Coon", "Miniature_Pinscher", "Newfoundland", 
        "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian_Blue", "Saint_Bernard", 
        "Samoyed", "Scottish_Terrier", "Shiba_Inu", "Siamese", "Sphynx", 
        "Staffordshire_Bull_Terrier", "Wheaten_Terrier", "Yorkshire_Terrier"
    ]
    
    label_path = os.path.join(output_dir, "label_list.txt")
    with open(label_path, "w") as f:
        for lab in label_list:
            f.write(lab + "\n")
    print(f"✅ Created {label_path} ({len(label_list)} classes)")

if __name__ == "__main__":
    # Ensure this points to where you extracted the 'images' and 'annotations' folders
    # Example structure:
    # dataset/oxford_pets/
    #   ├── images/
    #   ├── annotations/
    #       ├── xmls/
    #       ├── trainval.txt
    #       └── test.txt
    
    target_dir = "dataset/pets"
    
    if os.path.exists(target_dir):
        create_pets_filelists(target_dir)
    else:
        print(f"❌ Error: Directory {target_dir} not found. Have you downloaded the dataset?")