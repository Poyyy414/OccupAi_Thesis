# OccupAI Web Dashboard + Flask API

## Setup

1. Install dependencies:
```
pip install flask flask-cors ultralytics opencv-python numpy
```

2. Make sure your webcam is plugged in

3. Run the server:
```
python app.py
```

4. Open browser:
```
http://localhost:5000
```

## API Endpoints (for Flutter)

| Endpoint | Description |
|---|---|
| GET /api/stats | occupied, free, total, % |
| GET /api/occupancy | full slot data |
| GET /api/snapshot | camera frame (base64) |
| GET /api/history | occupancy history |
| GET /api/predictions | peak hours |
| POST /api/slots | save slots |
| POST /api/slots/auto | auto generate slots |

## Flutter Usage

```dart
final res = await http.get(Uri.parse('http://YOUR_IP:5000/api/stats'));
final data = jsonDecode(res.body);
print(data['free']); // available slots
```

Replace YOUR_IP with your laptop's local IP address.
Find it with: ipconfig (Windows) or ifconfig (Mac/Linux)
