import numpy as np
from sklearn.cluster import DBSCAN

class SpatialTemporalPredictor:
    def __init__(self, eps_km=1.5, min_samples=2):
        # 1.5 km spatial radius converted to radian distance
        self.kms_per_radian = 6371.0088
        self.epsilon = eps_km / self.kms_per_radian
        self.min_samples = min_samples

    def detect_active_hotspots(self, complaints: list):
        if len(complaints) < self.min_samples:
            return []

        coords = np.array([[c["suspected_atm_lat"], c["suspected_atm_lon"]] for c in complaints])
        radians_coords = np.radians(coords)

        db = DBSCAN(eps=self.epsilon, min_samples=self.min_samples, metric='haversine')
        cluster_labels = db.fit_predict(radians_coords)

        hotspots = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue  # Skip noise

            indices = np.where(cluster_labels == cluster_id)[0]
            cluster_points = coords[indices]
            total_at_risk = sum(complaints[i]["amount_lost"] for i in indices)
            
            centroid_lat = float(np.mean(cluster_points[:, 0]))
            centroid_lon = float(np.mean(cluster_points[:, 1]))

            hotspots.append({
                "hotspot_id": f"HS-CLUSTER-{cluster_id}",
                "latitude": centroid_lat,
                "longitude": centroid_lon,
                "active_complaint_count": len(indices),
                "total_funds_at_risk": total_at_risk,
                "threat_level": "RED" if total_at_risk > 100000 else "AMBER"
            })

        return hotspots