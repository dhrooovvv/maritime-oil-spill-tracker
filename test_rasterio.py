import rasterio

# Path to the GeoTIFF file we downloaded
geotiff_path = "sample_sar.tif"

# Open file with rasterio
with rasterio.open(geotiff_path) as dataset:
    print("--- GeoTIFF Metadata Successfully Read ---")
    print(f"Driver/Format : {dataset.driver}")
    print(f"Width x Height : {dataset.width} x {dataset.height} pixels")
    print(f"Number of Bands: {dataset.count}")
    print(f"Coordinate Ref System (CRS): {dataset.crs}")
    
    # Extract Bounding Box (Min Lon, Min Lat, Max Lon, Max Lat)
    bounds = dataset.bounds
    print("\n--- Map Bounding Box ---")
    print(f"West (Min Lon): {bounds.left}")
    print(f"South (Min Lat): {bounds.bottom}")
    print(f"East (Max Lon): {bounds.right}")
    print(f"North (Max Lat): {bounds.top}")

    # Extract exact real GPS coordinate of center pixel (X=width/2, Y=height/2)
    center_x = dataset.width // 2
    center_y = dataset.height // 2
    lon, lat = dataset.xy(center_y, center_x)
    
    print("\n🎯 Center Pixel GPS Calculation:")
    print(f"Pixel Position: ({center_x}, {center_y})")
    print(f"Calculated GPS: Latitude {lat:.4f}, Longitude {lon:.4f}")