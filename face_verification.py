from deepface import DeepFace


def verify_faces(reference_image, test_image):
    try:
        result = DeepFace.verify(
            img1_path=reference_image,
            img2_path=test_image,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True
        )

        verified = result["verified"]
        distance = result["distance"]

        return verified, distance

    except Exception as e:
        print(f"Verification error: {e}")
        raise


if __name__ == "__main__":
    reference_image = "reference.jpg"
    test_image = "test.jpg"

    verified, distance = verify_faces(
        reference_image,
        test_image
    )

    print("\n===== Face Verification Result =====")

    if verified:
        print("Result: MATCH")
    else:
        print("Result: NOT MATCH")

    if distance is not None:
        print(f"Distance: {distance:.4f}")