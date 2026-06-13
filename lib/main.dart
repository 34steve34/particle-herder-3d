import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:math' as math;
import 'dart:async';
import 'dart:ui' as ui;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vector_math/vector_math_64.dart' as vm;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Changed to lock the game into landscape orientation
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  runApp(const ParticleHerderApp());
}

class ParticleHerderApp extends StatelessWidget {
  const ParticleHerderApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Particle Herder 3D',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const GameScreen(),
    );
  }
}

// ==========================================
// 1. DATA STRUCTURES & SPATIAL PARTITIONING
// ==========================================

enum ParticleState { incubating, active }

class Particle3D {
  vm.Vector3 position;
  vm.Vector3 velocity;
  ParticleState state;
  double radius;
  Color color;

  Particle3D({
    required this.position,
    required this.velocity,
    this.state = ParticleState.incubating,
    this.radius = 3.0,
    this.color = const Color(0xFF00FF00),
  });
}

class GravityImpulse {
  final vm.Vector3 position;
  final DateTime createdAt;
  final double maxRadius = 150.0;
  final Duration duration = const Duration(milliseconds: 600);

  GravityImpulse({required this.position}) : createdAt = DateTime.now();

  double getProgress() {
    final elapsed = DateTime.now().difference(createdAt).inMilliseconds;
    return (elapsed / duration.inMilliseconds).clamp(0.0, 1.0);
  }
}

class AABB {
  final vm.Vector3 min;
  final vm.Vector3 max;
  AABB(this.min, this.max);

  bool contains(vm.Vector3 point) {
    return point.x >= min.x && point.x <= max.x &&
           point.y >= min.y && point.y <= max.y &&
           point.z >= min.z && point.z <= max.z;
  }

  bool intersects(AABB other) {
    return (min.x <= other.max.x && max.x >= other.min.x) &&
           (min.y <= other.max.y && max.y >= other.min.y) &&
           (min.z <= other.max.z && max.z >= other.min.z);
  }
}

class Octree {
  final AABB boundary;
  final int capacity;
  final List<Particle3D> particles = [];
  bool isDivided = false;

  late Octree topLeftFront, topRightFront, bottomLeftFront, bottomRightFront;
  late Octree topLeftBack, topRightBack, bottomLeftBack, bottomRightBack;

  Octree(this.boundary, {this.capacity = 8});

  void subdivide() {
    final cx = (boundary.min.x + boundary.max.x) / 2;
    final cy = (boundary.min.y + boundary.max.y) / 2;
    final cz = (boundary.min.z + boundary.max.z) / 2;

    topLeftFront = Octree(AABB(vm.Vector3(boundary.min.x, boundary.min.y, boundary.min.z), vm.Vector3(cx, cy, cz)), capacity: capacity);
    topRightFront = Octree(AABB(vm.Vector3(cx, boundary.min.y, boundary.min.z), vm.Vector3(boundary.max.x, cy, cz)), capacity: capacity);
    bottomLeftFront = Octree(AABB(vm.Vector3(boundary.min.x, cy, boundary.min.z), vm.Vector3(cx, boundary.max.y, cz)), capacity: capacity);
    bottomRightFront = Octree(AABB(vm.Vector3(cx, cy, boundary.min.z), vm.Vector3(boundary.max.x, boundary.max.y, cz)), capacity: capacity);

    topLeftBack = Octree(AABB(vm.Vector3(boundary.min.x, boundary.min.y, cz), vm.Vector3(cx, cy, boundary.max.z)), capacity: capacity);
    topRightBack = Octree(AABB(vm.Vector3(cx, boundary.min.y, cz), vm.Vector3(boundary.max.x, cy, boundary.max.z)), capacity: capacity);
    bottomLeftBack = Octree(AABB(vm.Vector3(boundary.min.x, cy, cz), vm.Vector3(cx, boundary.max.y, boundary.max.z)), capacity: capacity);
    bottomRightBack = Octree(AABB(vm.Vector3(cx, cy, cz), vm.Vector3(boundary.max.x, boundary.max.y, boundary.max.z)), capacity: capacity);

    isDivided = true;
  }

  bool insert(Particle3D particle) {
    if (!boundary.contains(particle.position)) return false;

    if (particles.length < capacity && !isDivided) {
      particles.add(particle);
      return true;
    }

    if (!isDivided) subdivide();

    return topLeftFront.insert(particle) || topRightFront.insert(particle) ||
           bottomLeftFront.insert(particle) || bottomRightFront.insert(particle) ||
           topLeftBack.insert(particle) || topRightBack.insert(particle) ||
           bottomLeftBack.insert(particle) || bottomRightBack.insert(particle);
  }

  List<Particle3D> queryRange(AABB range) {
    List<Particle3D> found = [];
    if (!boundary.intersects(range)) return found;

    for (var p in particles) {
      if (range.contains(p.position)) found.add(p);
    }

    if (isDivided) {
      found.addAll(topLeftFront.queryRange(range));
      found.addAll(topRightFront.queryRange(range));
      found.addAll(bottomLeftFront.queryRange(range));
      found.addAll(bottomRightFront.queryRange(range));
      found.addAll(topLeftBack.queryRange(range));
      found.addAll(topRightBack.queryRange(range));
      found.addAll(bottomLeftBack.queryRange(range));
      found.addAll(bottomRightBack.queryRange(range));
    }
    return found;
  }
}

// ==========================================
// 2. RAYCASTING INTERSECTION SYSTEM
// ==========================================

class Ray {
  final vm.Vector3 origin;
  final vm.Vector3 direction;
  Ray(this.origin, this.direction) {
    direction.normalize();
  }
}

class Plane3D {
  final vm.Vector3 normal;
  final vm.Vector3 point;
  Plane3D(this.normal, this.point) {
    normal.normalize();
  }

  vm.Vector3? intersectRay(Ray ray) {
    double denom = normal.dot(ray.direction);
    if (denom.abs() > 1e-6) {
      vm.Vector3 p0minusO = point - ray.origin;
      double t = p0minusO.dot(normal) / denom;
      return ray.origin + (ray.direction * t);
    }
    return null;
  }
}

// ==========================================
// 3. MAIN GAME SCREEN
// ==========================================

class GameScreen extends StatefulWidget {
  const GameScreen({Key? key}) : super(key: key);

  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> with TickerProviderStateMixin {
  final vm.Vector3 boxDimensions = vm.Vector3(120.0, 120.0, 300.0);
  final double safeIncubationRadius = 25.0;

  bool isPlaying = false;
  int currentScore = 0;
  List<int> highScores = [];
  List<Particle3D> particles = [];
  List<GravityImpulse> activeImpulses = [];

  vm.Vector3? activeRayStart;
  vm.Vector3? activeRayEnd;
  double activeRayProgress = 0.0;
  int? activeRayPointerId;

  late Timer gameTimer;
  late Timer spawnTimer;
  DateTime? gameStartTime;
  int elapsedMilliseconds = 0;

  double cameraRadius = 400.0;
  double cameraTheta = 0.78;
  double cameraPhi = 1.2;
  
  double autoRotateSpeedTheta = 0.12;
  double autoRotateSpeedPhi = 0.05;
  bool isUserInteractingWithBox = false;

  Map<int, Offset> activeTouches = {};
  int? cameraTrackingPointerId;

  late AnimationController _gameLoopController;

  @override
  void initState() {
    super.initState();
    _loadHighScores();
    _gameLoopController = AnimationController(vsync: this, duration: const Duration(days: 1))..addListener(_updatePhysicsLoop);
  }

  @override
  void dispose() {
    _gameLoopController.dispose();
    if (isPlaying) {
      gameTimer.cancel();
      spawnTimer.cancel();
    }
    super.dispose();
  }

  Future<void> _loadHighScores() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      highScores = prefs.getStringList('highScores3D')?.map(int.parse).toList() ?? [];
      highScores.sort((a, b) => b.compareTo(a));
    });
  }

  Future<void> _saveHighScore(int score) async {
    final prefs = await SharedPreferences.getInstance();
    highScores.add(score);
    highScores.sort((a, b) => b.compareTo(a));
    if (highScores.length > 5) highScores = highScores.sublist(0, 5);
    await prefs.setStringList('highScores3D', highScores.map((e) => e.toString()).toList());
    setState(() {});
  }

  void _startNewGame() {
    setState(() {
      particles.clear();
      activeImpulses.clear();
      activeRayStart = null;
      activeRayEnd = null;
      activeRayProgress = 0.0;
      activeRayPointerId = null;
      currentScore = 0;
      elapsedMilliseconds = 0;
      isPlaying = true;
      gameStartTime = DateTime.now();
      _spawnParticle();
    });

    gameTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      if (!isPlaying) return;
      setState(() {
        elapsedMilliseconds = DateTime.now().difference(gameStartTime!).inMilliseconds;
        currentScore = elapsedMilliseconds ~/ 100;
      });
    });

    spawnTimer = Timer.periodic(const Duration(milliseconds: 800), (timer) {
      if (isPlaying) _spawnParticle();
    });

    _gameLoopController.repeat();
  }

  void _triggerGameOver() {
    if (!isPlaying) return;
    setState(() {
      isPlaying = false;
    });
    _gameLoopController.stop();
    gameTimer.cancel();
    spawnTimer.cancel();
    _saveHighScore(currentScore);
  }

  void _spawnParticle() {
    final p = Particle3D(
      position: vm.Vector3(0, 0, 0),
      velocity: vm.Vector3(
        (math.Random().nextDouble() * 2 - 1) * 5,
        (math.Random().nextDouble() * 2 - 1) * 5,
        (math.Random().nextDouble() * 2 - 1) * 9,
      ),
      state: ParticleState.incubating,
      color: const Color(0xFF00FF00),
    );
    particles.add(p);
  }

  void _updatePhysicsLoop() {
    if (!isPlaying) return;

    double dt = 0.016;

    if (activeRayStart != null) {
      setState(() {
        activeRayProgress += dt * 1.8;
        if (activeRayProgress > 1.0) activeRayProgress = 1.0;
      });
    }

    if (!isUserInteractingWithBox) {
      setState(() {
        cameraTheta += autoRotateSpeedTheta * dt;
        cameraPhi = (cameraPhi + autoRotateSpeedPhi * dt).clamp(0.2, math.pi - 0.2);
      });
    }

    final halfX = boxDimensions.x / 2;
    final halfY = boxDimensions.y / 2;
    final halfZ = boxDimensions.z / 2;
    AABB spatialVolume = AABB(vm.Vector3(-halfX, -halfY, -halfZ), vm.Vector3(halfX, halfY, halfZ));
    Octree frameOctree = Octree(spatialVolume);

    activeImpulses.removeWhere((impulse) => impulse.getProgress() >= 1.0);

    for (var p in particles) {
      frameOctree.insert(p);
    }

    setState(() {
      for (var p in particles) {
        double distanceToCenter = p.position.length;

        if (p.state == ParticleState.incubating) {
          if (distanceToCenter > safeIncubationRadius) {
            p.state = ParticleState.active;
          } else {
            p.position += p.velocity * dt * 0.4;
          }
        }

        if (p.state == ParticleState.active) {
          // Strictly apply constant velocity. No centrifugal ghost acceleration.
          p.position += p.velocity * dt;

          double threatFactor = (p.position.xy.length / halfX).clamp(0.0, 1.0);
          if (threatFactor < 0.5) {
            p.color = Color.lerp(const Color(0xFF00FF00), const Color(0xFFFFA500), threatFactor * 2)!;
          } else {
            p.color = Color.lerp(const Color(0xFFFFA500), const Color(0xFFFF0000), (threatFactor - 0.5) * 2)!;
          }
        }

        for (var impulse in activeImpulses) {
          double distanceToImpulse = (p.position - impulse.position).length;
          double currentRadius = impulse.maxRadius * impulse.getProgress();
          
          if (distanceToImpulse < currentRadius && distanceToImpulse > 5.0) {
            double originalSpeed = p.velocity.length;
            
            vm.Vector3 forceDirection = impulse.position - p.position;
            forceDirection.normalize();
            double pullIntensity = (1.0 - (distanceToImpulse / impulse.maxRadius)) * 140.0;
            
            p.velocity += forceDirection * pullIntensity * dt;
            
            // Restore original speed magnitude - only direction changes
            p.velocity.normalize();
            p.velocity *= originalSpeed;
          }
        }

        if (p.position.z.abs() > halfZ) {
          p.position.z = -p.position.z;
        }

        if (p.position.x.abs() >= halfX || p.position.y.abs() >= halfY) {
          _triggerGameOver();
          break;
        }
      }
    });
  }

  vm.Vector3 _computeCameraPosition() {
    double x = cameraRadius * math.sin(cameraPhi) * math.cos(cameraTheta);
    double y = cameraRadius * math.cos(cameraPhi);
    double z = cameraRadius * math.sin(cameraPhi) * math.sin(cameraTheta);
    return vm.Vector3(x, y, z);
  }

  Ray _castScreenRay(Offset touchPoint, Size widgetBounds) {
    vm.Vector3 camPos = _computeCameraPosition();
    vm.Vector3 target = vm.Vector3(0, 0, 0);
    vm.Vector3 up = vm.Vector3(0, 1, 0);

    vm.Matrix4 viewMatrix = vm.makeViewMatrix(camPos, target, up);
    vm.Matrix4 projectionMatrix = vm.makePerspectiveMatrix(vm.radians(45.0), widgetBounds.width / widgetBounds.height, 10.0, 1000.0);
    vm.Matrix4 inverseProjectionView = vm.Matrix4.copy(projectionMatrix * viewMatrix)..invert();

    double ndcX = (touchPoint.dx / widgetBounds.width) * 2.0 - 1.0;
    double ndcY = 1.0 - (touchPoint.dy / widgetBounds.height) * 2.0;

    vm.Vector4 nearPoint = vm.Vector4(ndcX, ndcY, -1.0, 1.0);
    vm.Vector4 farPoint = vm.Vector4(ndcX, ndcY, 1.0, 1.0);

    vm.Vector4 worldNear = inverseProjectionView * nearPoint;
    vm.Vector4 worldFar = inverseProjectionView * farPoint;

    worldNear.scale(1.0 / worldNear.w);
    worldFar.scale(1.0 / worldFar.w);

    vm.Vector3 rayOrigin = vm.Vector3(worldNear.x, worldNear.y, worldNear.z);
    vm.Vector3 rayDirection = vm.Vector3(worldFar.x - worldNear.x, worldFar.y - worldNear.y, worldFar.z - worldNear.z)..normalize();

    return Ray(rayOrigin, rayDirection);
  }

  List<vm.Vector3> _findRayBoxIntersections(Ray ray) {
    double halfX = boxDimensions.x / 2;
    double halfY = boxDimensions.y / 2;
    double halfZ = boxDimensions.z / 2;

    List<Plane3D> planes = [
      Plane3D(vm.Vector3(1, 0, 0), vm.Vector3(halfX, 0, 0)),
      Plane3D(vm.Vector3(-1, 0, 0), vm.Vector3(-halfX, 0, 0)),
      Plane3D(vm.Vector3(0, 1, 0), vm.Vector3(0, halfY, 0)),
      Plane3D(vm.Vector3(0, -1, 0), vm.Vector3(0, -halfY, 0)),
      Plane3D(vm.Vector3(0, 0, 1), vm.Vector3(0, 0, halfZ)),
      Plane3D(vm.Vector3(0, 0, -1), vm.Vector3(0, 0, -halfZ)),
    ];

    List<MapEntry<vm.Vector3, double>> crossPoints = [];

    for (var plane in planes) {
      vm.Vector3? pt = plane.intersectRay(ray);
      if (pt != null) {
        if (pt.x.abs() <= halfX + 0.1 && pt.y.abs() <= halfY + 0.1 && pt.z.abs() <= halfZ + 0.1) {
          double dist = (pt - ray.origin).length;
          crossPoints.add(MapEntry(pt, dist));
        }
      }
    }

    if (crossPoints.length >= 2) {
      crossPoints.sort((a, b) => a.value.compareTo(b.value));
      return [crossPoints.first.key, crossPoints.last.key];
    }
    return crossPoints.map((e) => e.key).toList();
  }

  void _handleTouchDown(int pointerId, Offset localPosition, Size screenSize) {
    double marginX = screenSize.width * 0.15;
    double marginY = screenSize.height * 0.15;
    bool insideOutskirtsZone = localPosition.dx < marginX || 
                               localPosition.dx > screenSize.width - marginX ||
                               localPosition.dy < marginY || 
                               localPosition.dy > screenSize.height - marginY;

    if (insideOutskirtsZone && cameraTrackingPointerId == null) {
      cameraTrackingPointerId = pointerId;
      activeTouches[pointerId] = localPosition;
    } else if (!insideOutskirtsZone) {
      isUserInteractingWithBox = true;

      if (activeTouches.isEmpty) {
        // First touch - create ray
        Ray ray = _castScreenRay(localPosition, screenSize);
        List<vm.Vector3> targets = _findRayBoxIntersections(ray);
        if (targets.length >= 2) {
          setState(() {
            activeRayStart = targets[0];
            activeRayEnd = targets[1];
            activeRayProgress = 0.0;
            activeRayPointerId = pointerId;
          });
        }
      } else if (activeTouches.length >= 1) {
        // Second (or more) touch - trigger multi-touch intersection
        setState(() {
          activeRayStart = null;
          activeRayEnd = null;
          activeRayPointerId = null;
        });
        _processMultiTouchIntersection(screenSize);
      }
      
      activeTouches[pointerId] = localPosition;
    }
  }

  void _handleTouchMove(int pointerId, Offset localPosition, Size screenSize) {
    if (!activeTouches.containsKey(pointerId)) return;
    Offset previousPosition = activeTouches[pointerId]!;
    activeTouches[pointerId] = localPosition;

    if (pointerId == cameraTrackingPointerId) {
      Offset delta = localPosition - previousPosition;
      setState(() {
        cameraTheta -= delta.dx * 0.007;
        cameraPhi = (cameraPhi - delta.dy * 0.007).clamp(0.2, math.pi - 0.2);
      });
    }
  }

  void _handleTouchUp(int pointerId, Size screenSize) {
    if (pointerId == cameraTrackingPointerId) {
      cameraTrackingPointerId = null;
    } else if (pointerId == activeRayPointerId) {
      if (activeRayStart != null && activeRayEnd != null) {
        vm.Vector3 finalDropPoint = activeRayStart! + (activeRayEnd! - activeRayStart!) * activeRayProgress;
        setState(() {
          activeImpulses.add(GravityImpulse(position: finalDropPoint));
        });
      }
      setState(() {
        activeRayStart = null;
        activeRayEnd = null;
        activeRayPointerId = null;
      });
    }

    activeTouches.remove(pointerId);
    if (activeTouches.isEmpty) {
      isUserInteractingWithBox = false;
    }
  }

  void _processMultiTouchIntersection(Size bounds) {
    if (activeTouches.length < 2) return;
    var keys = activeTouches.keys.toList();
    
    Ray ray1 = _castScreenRay(activeTouches[keys[0]]!, bounds);
    Ray ray2 = _castScreenRay(activeTouches[keys[1]]!, bounds);

    List<vm.Vector3> pts1 = _findRayBoxIntersections(ray1);
    List<vm.Vector3> pts2 = _findRayBoxIntersections(ray2);

    if (pts1.isNotEmpty && pts2.isNotEmpty) {
      vm.Vector3 p1 = pts1.first;
      vm.Vector3 p2 = pts2.first;
      vm.Vector3 lineVec = p2 - p1;
      double midpointFactor = lineVec.length * 0.5;
      vm.Vector3 midPointIntersection = p1 + (lineVec.normalized() * midpointFactor);

      setState(() {
        activeImpulses.add(GravityImpulse(position: midPointIntersection));
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final Size screenSize = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: const Color(0xFF020208),
      body: Stack(
        children: [
          Listener(
            onPointerDown: (e) => _handleTouchDown(e.pointer, e.localPosition, screenSize),
            onPointerMove: (e) => _handleTouchMove(e.pointer, e.localPosition, screenSize),
            onPointerUp: (e) => _handleTouchUp(e.pointer, screenSize),
            onPointerCancel: (e) => _handleTouchUp(e.pointer, screenSize),
            child: CustomPaint(
              size: Size.infinite,
              painter: Scene3DPainter(
                boxDimensions: boxDimensions,
                particles: particles,
                impulses: activeImpulses,
                cameraPosition: _computeCameraPosition(),
                safeRadius: safeIncubationRadius,
                activeRayStart: activeRayStart,
                activeRayEnd: activeRayEnd,
                activeRayProgress: activeRayProgress,
              ),
            ),
          ),

          // Cache-busting Version Indicator
          const Positioned(
            top: 10.0,
            left: 10.0,
            child: Text(
              'v2.7.1-STRICT-PHYSICS',
              style: TextStyle(
                color: Colors.cyanAccent,
                fontSize: 12,
                fontFamily: 'monospace',
                fontWeight: FontWeight.bold,
              ),
            ),
          ),

          Positioned(
            top: 40.0,
            left: 20.0,
            child: Text(
              'SCORE: $currentScore',
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                letterSpacing: 2,
                color: Colors.white,
              ),
            ),
          ),

          if (!isPlaying) _buildMenuOverlay(),
        ],
      ),
    );
  }

  Widget _buildMenuOverlay() {
    return Container(
      color: Colors.black87,
      child: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                'PARTICLE HERDER 3D',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.w900, letterSpacing: 3, color: Colors.green),
              ),
              const SizedBox(height: 15),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40.0),
                child: Text(
                  'Herd particles away from the 4 long side walls.\n'
                  'Square ends wrap around continuously.\n\n'
                  '• Swipe Outskirts to rotate framework camera\n'
                  '• Hold Box to charge a Rainbow Depth Ray (Violet = Center)\n'
                  '• Touch 2 faces to drop a Crosshair Intersection',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.grey[400], fontSize: 13, height: 1.5),
                ),
              ),
              const SizedBox(height: 40),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                ),
                onPressed: _startNewGame,
                child: const Text('START HERDING', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.black)),
              ),
              if (highScores.isNotEmpty) ...[
                const SizedBox(height: 40),
                const Text('TOP SCORING RUNS', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, letterSpacing: 1)),
                const SizedBox(height: 10),
                ...highScores.asMap().entries.map((entry) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 3.0),
                  child: Text('${entry.key + 1}. ${entry.value}', style: const TextStyle(fontSize: 14, color: Colors.grey)),
                )),
              ]
            ],
          ),
        ),
      ),
    );
  }
}

// ==========================================
// 8. GRAPHICS PIPELINE & MATH PAINTER
// ==========================================

class Scene3DPainter extends CustomPainter {
  final vm.Vector3 boxDimensions;
  final List<Particle3D> particles;
  final List<GravityImpulse> impulses;
  final vm.Vector3 cameraPosition;
  final double safeRadius;

  final vm.Vector3? activeRayStart;
  final vm.Vector3? activeRayEnd;
  final double activeRayProgress;

  Scene3DPainter({
    required this.boxDimensions,
    required this.particles,
    required this.impulses,
    required this.cameraPosition,
    required this.safeRadius,
    this.activeRayStart,
    this.activeRayEnd,
    required this.activeRayProgress,
  });

  @override
  void paint(Canvas canvas, Size size) {
    vm.Vector3 target = vm.Vector3(0, 0, 0);
    vm.Vector3 up = vm.Vector3(0, 1, 0);

    vm.Matrix4 viewMatrix = vm.makeViewMatrix(cameraPosition, target, up);
    vm.Matrix4 projectionMatrix = vm.makePerspectiveMatrix(vm.radians(45.0), size.width / size.height, 10.0, 1000.0);
    vm.Matrix4 vpMatrix = projectionMatrix * viewMatrix;

    _drawBoundingBox(canvas, size, vpMatrix);
    _drawGravityImpulses(canvas, size, vpMatrix);
    _drawColorCodedRay(canvas, size, vpMatrix);
    _drawParticles(canvas, size, vpMatrix);
  }

  Offset? _projectPoint(vm.Vector3 point, Size size, vm.Matrix4 vpMatrix) {
    vm.Vector4 pos4 = vm.Vector4(point.x, point.y, point.z, 1.0);
    vm.Vector4 projected = vpMatrix * pos4;

    if (projected.w <= 0) return null;

    double x = (projected.x / projected.w + 1.0) * size.width / 2.0;
    double y = (1.0 - projected.y / projected.w) * size.height / 2.0;
    return Offset(x, y);
  }

  void _drawColorCodedRay(Canvas canvas, Size size, vm.Matrix4 vpMatrix) {
    if (activeRayStart == null || activeRayEnd == null) return;

    Offset? pStart = _projectPoint(activeRayStart!, size, vpMatrix);
    Offset? pEnd = _projectPoint(activeRayEnd!, size, vpMatrix);

    vm.Vector3 currentTipVec = activeRayStart! + (activeRayEnd! - activeRayStart!) * activeRayProgress;
    Offset? pTip = _projectPoint(currentTipVec, size, vpMatrix);

    if (pStart != null && pEnd != null && pTip != null) {
      final List<Color> spectrumColors = [
        Colors.red, Colors.orange, Colors.yellow, Colors.green,
        Colors.blue, Colors.indigo, Colors.purple, Colors.indigo,
        Colors.blue, Colors.green, Colors.yellow, Colors.orange, Colors.red,
      ];

      final Paint clearFallbackPaint = Paint()
        ..color = Colors.white.withOpacity(0.35)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5;
      canvas.drawLine(pStart, pTip, clearFallbackPaint);

      final Paint rayPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4.0
        ..isAntiAlias = true
        ..shader = ui.Gradient.linear(pStart, pEnd, spectrumColors);
      canvas.drawLine(pStart, pTip, rayPaint);

      final Paint tipCorePaint = Paint()
        ..color = Colors.white
        ..style = PaintingStyle.fill;
      canvas.drawCircle(pTip, 4.5, tipCorePaint);

      final Paint tipGlowPaint = Paint()
        ..color = Colors.purple.withOpacity(0.4)
        ..style = PaintingStyle.fill;
      canvas.drawCircle(pTip, 9.0, tipGlowPaint);
    }
  }

  void _drawBoundingBox(Canvas canvas, Size size, vm.Matrix4 vpMatrix) {
    double hX = boxDimensions.x / 2;
    double hY = boxDimensions.y / 2;
    double hZ = boxDimensions.z / 2;

    List<vm.Vector3> vertices = [
      vm.Vector3(-hX, -hY, -hZ), vm.Vector3(hX, -hY, -hZ),
      vm.Vector3(hX, hY, -hZ), vm.Vector3(-hX, hY, -hZ),
      vm.Vector3(-hX, -hY, hZ), vm.Vector3(hX, -hY, hZ),
      vm.Vector3(hX, hY, hZ), vm.Vector3(-hX, hY, hZ),
    ];

    List<Offset?> projected = vertices.map((v) => _projectPoint(v, size, vpMatrix)).toList();

    List<List<int>> edges = [
      [0, 1], [1, 2], [2, 3], [3, 0], 
      [4, 5], [5, 6], [6, 7], [7, 4], 
      [0, 4], [1, 5], [2, 6], [3, 7], 
    ];

    final Paint edgePaint = Paint()
      ..style = PaintingStyle.stroke
      ..isAntiAlias = true;

    for (var edge in edges) {
      Offset? p1 = projected[edge[0]];
      Offset? p2 = projected[edge[1]];

      if (p1 != null && p2 != null) {
        double averageDepth = (vertices[edge[0]].z + vertices[edge[1]].z) / (hZ * 2) + 0.5;
        
        edgePaint.color = Colors.cyan.withOpacity(math.max(0.15, 1.0 - averageDepth));
        edgePaint.strokeWidth = math.max(1.0, 3.5 * (1.0 - averageDepth));

        canvas.drawLine(p1, p2, edgePaint);
      }
    }

    _drawInteriorSubGrids(canvas, size, vpMatrix, hX, hY, hZ);
  }

  void _drawInteriorSubGrids(Canvas canvas, Size size, vm.Matrix4 vpMatrix, double hX, double hY, double hZ) {
    final gridPaint = Paint()
      ..color = Colors.cyan.withOpacity(0.06)
      ..strokeWidth = 0.8
      ..style = PaintingStyle.stroke;

    for (double i = -hZ + 50; i < hZ; i += 50) {
      List<vm.Vector3> ring = [
        vm.Vector3(-hX, -hY, i), vm.Vector3(hX, -hY, i),
        vm.Vector3(hX, hY, i), vm.Vector3(-hX, hY, i)
      ];
      List<Offset?> projRing = ring.map((v) => _projectPoint(v, size, vpMatrix)).toList();
      for (int j = 0; j < 4; j++) {
        Offset? p1 = projRing[j];
        Offset? p2 = projRing[(j + 1) % 4];
        if (p1 != null && p2 != null) canvas.drawLine(p1, p2, gridPaint);
      }
    }
  }

  void _drawParticles(Canvas canvas, Size size, vm.Matrix4 vpMatrix) {
    final Paint pPaint = Paint()..blendMode = BlendMode.plus;

    for (var p in particles) {
      Offset? screenPos = _projectPoint(p.position, size, vpMatrix);
      if (screenPos == null) continue;

      double distanceToCam = (cameraPosition - p.position).length;
      double scale = (450.0 / distanceToCam).clamp(0.3, 3.0);

      double baseRadius = p.state == ParticleState.incubating ? p.radius * 0.7 : p.radius;
      double finalRadius = baseRadius * scale;

      pPaint.color = p.color.withOpacity((0.9 * (scale / 3.0)).clamp(0.2, 1.0));

      canvas.drawCircle(screenPos, finalRadius, pPaint);

      pPaint.color = p.color.withOpacity(0.25);
      canvas.drawCircle(screenPos, finalRadius * 2.2, pPaint);

      double halfZ = boxDimensions.z / 2;
      if (p.position.z.abs() > halfZ * 0.75) {
        vm.Vector3 ghostPosition = vm.Vector3(p.position.x, p.position.y, -p.position.z);
        Offset? ghostScreenPos = _projectPoint(ghostPosition, size, vpMatrix);
        if (ghostScreenPos != null) {
          final Paint ghostPaint = Paint()
            ..color = p.color.withOpacity(0.08)
            ..blendMode = BlendMode.plus;
          canvas.drawCircle(ghostScreenPos, finalRadius * 0.8, ghostPaint);
        }
      }
    }
  }

  void _drawGravityImpulses(Canvas canvas, Size size, vm.Matrix4 vpMatrix) {
    final Paint pulsePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    for (var impulse in impulses) {
      Offset? screenPos = _projectPoint(impulse.position, size, vpMatrix);
      if (screenPos == null) continue;

      double progress = impulse.getProgress();
      double radius = impulse.maxRadius * progress;

      pulsePaint.color = Colors.white.withOpacity(1.0 - progress);
      canvas.drawCircle(screenPos, radius, pulsePaint);
    }
  }

  @override
  bool shouldRepaint(covariant Scene3DPainter oldDelegate) => true;
}
