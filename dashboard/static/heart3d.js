/**
 * heart3d.js — Realistic Anatomical 3D Human Heart for Precision Cardiology
 * =========================================================================
 * Loads and renders a high-resolution anatomical ZBrush medical sculpt
 * with photographic PBR texture maps, physiological torsional wringing systole,
 * dynamic ischemic hypoperfusion shading, and clinical viewpoint controls.
 */

(function () {
  "use strict";

  /* ───────────────────────── INLINE ORBIT CONTROLS ────────────────── */
  function buildOrbitControls(THREE) {
    function OrbitControls(camera, domElement) {
      this.camera = camera;
      this.domElement = domElement;
      this.enabled = true;
      this.target = new THREE.Vector3(0, 0, 0);
      this.minDistance = 3.8;
      this.maxDistance = 14.0;
      this.enableDamping = true;
      this.dampingFactor = 0.08;
      this.autoRotate = true;
      this.autoRotateSpeed = 0.65;
      this.enableZoom = true;

      let _spherical = new THREE.Spherical();
      let _sphericalDelta = new THREE.Spherical();
      let _scale = 1;
      let _isPointerDown = false;
      let _rotateStart = new THREE.Vector2();
      let _rotateDelta = new THREE.Vector2();

      const offset = new THREE.Vector3();
      offset.copy(camera.position).sub(this.target);
      _spherical.setFromVector3(offset);

      this.setCameraLook = function (radius, phi, theta) {
        _spherical.radius = radius;
        _spherical.phi = phi;
        _spherical.theta = theta;
        _sphericalDelta.set(0, 0, 0);
        _scale = 1;
      };

      this.update = () => {
        if (this.autoRotate && !_isPointerDown) {
          _sphericalDelta.theta -= ((2 * Math.PI) / 60 / 60) * this.autoRotateSpeed;
        }
        _spherical.theta += _sphericalDelta.theta;
        _spherical.phi += _sphericalDelta.phi;
        _spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, _spherical.phi));
        _spherical.radius *= _scale;
        _spherical.radius = Math.max(this.minDistance, Math.min(this.maxDistance, _spherical.radius));
        _scale = 1;

        if (this.enableDamping) {
          _sphericalDelta.theta *= 1 - this.dampingFactor;
          _sphericalDelta.phi   *= 1 - this.dampingFactor;
        } else {
          _sphericalDelta.set(0, 0, 0);
        }
        offset.setFromSpherical(_spherical);
        camera.position.copy(this.target).add(offset);
        camera.lookAt(this.target);
      };

      const onPointerDown = (e) => {
        if (!this.enabled) return;
        _isPointerDown = true;
        _rotateStart.set(e.clientX, e.clientY);
      };
      const onPointerMove = (e) => {
        if (!this.enabled || !_isPointerDown) return;
        _rotateDelta.set(e.clientX - _rotateStart.x, e.clientY - _rotateStart.y);
        const el = domElement;
        _sphericalDelta.theta -= (2 * Math.PI * _rotateDelta.x) / el.clientHeight;
        _sphericalDelta.phi   -= (2 * Math.PI * _rotateDelta.y) / el.clientHeight;
        _rotateStart.set(e.clientX, e.clientY);
      };
      const onPointerUp = () => { _isPointerDown = false; };
      const onWheel = (e) => {
        if (!this.enabled || !this.enableZoom) return;
        if (e.deltaY < 0) _scale *= 0.94;
        else _scale /= 0.94;
        e.preventDefault();
      };

      domElement.addEventListener("pointerdown", onPointerDown);
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
      domElement.addEventListener("wheel", onWheel, { passive: false });
    }
    return OrbitControls;
  }

  /* ───────────────────────── MAIN INITIALIZATION ──────────────────── */
  function initHeart(containerId) {
    const container = document.getElementById(containerId || "heart-container");
    if (!container) return;

    const THREE = window.THREE;
    if (!THREE) return;

    const OrbitControls = buildOrbitControls(THREE);

    /* --- High Precision Renderer --- */
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    /* --- Scene & Medical Perspective Camera --- */
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 0.4, 6.8);

    /* --- Studio & Surgical 3-Point Lighting --- */
    const hemiLight = new THREE.HemisphereLight(0x223048, 0x0c111c, 1.4);
    scene.add(hemiLight);

    // Key Light (Surgical lamp with soft shadows)
    const keyLight = new THREE.DirectionalLight(0xfff6ee, 2.6);
    keyLight.position.set(4, 7, 5.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.bias = -0.0008;
    scene.add(keyLight);

    // Cool Lateral Fill Light (reveals muscular sulci)
    const fillLight = new THREE.DirectionalLight(0x7aa0c8, 1.2);
    fillLight.position.set(-6, 2, 4);
    scene.add(fillLight);

    // Arterial Rim Backlight (accentuates coronary vessel contours)
    const rimLight = new THREE.PointLight(0xff6e7a, 1.6, 20);
    rimLight.position.set(-3, -2, -5);
    scene.add(rimLight);

    // Inferior Diaphragmatic Light
    const underLight = new THREE.PointLight(0x405570, 0.9, 12);
    underLight.position.set(1, -5, -2);
    scene.add(underLight);

    /* --- Heart Root Group --- */
    const heartGroup = new THREE.Group();
    scene.add(heartGroup);

    // Slight physiological chest tilt (~18° yaw, 10° pitch)
    heartGroup.rotation.set(0.08, -0.28, 0.15);

    /* --- OrbitControls --- */
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.55;

    /* --- Model References --- */
    const heartMeshes = [];
    const heartMaterials = [];

    // Helper: Convert Base64 to ArrayBuffer
    function b64ToArrayBuffer(base64) {
      const binaryString = window.atob(base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      return bytes.buffer;
    }

    // Load High-Resolution Anatomical GLB
    function loadPhotorealisticHeart() {
      const b64Data = window.__HEART_GLB_B64__;
      if (!b64Data || !THREE.GLTFLoader) {
        console.warn("heart3d: GLB data or GLTFLoader not ready.");
        return;
      }

      const loader = new THREE.GLTFLoader();
      try {
        const buffer = b64ToArrayBuffer(b64Data);
        loader.parse(buffer, "", function (gltf) {
          const model = gltf.scene;

          // Compute exact bounding box and normalize scale & center
          const box = new THREE.Box3().setFromObject(model);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const targetScale = 3.6 / maxDim;

          model.scale.set(targetScale, targetScale, targetScale);
          model.position.set(
            -center.x * targetScale,
            -center.y * targetScale,
            -center.z * targetScale
          );

          // Configure realistic PBR tissue shaders
          model.traverse((child) => {
            if (child.isMesh) {
              child.castShadow = true;
              child.receiveShadow = true;
              heartMeshes.push(child);

              if (child.material) {
                // Ensure biological surface reflectivity (moist epicardium, non-plastic)
                child.material.roughness = 0.40;
                child.material.metalness = 0.04;
                child.material.envMapIntensity = 1.0;
                child.material.needsUpdate = true;
                heartMaterials.push(child.material);
              }
            }
          });

          heartGroup.add(model);

          // Hide loading spinner if present
          const spinner = document.getElementById("loading-scrim");
          if (spinner) {
            spinner.style.opacity = "0";
            setTimeout(() => { spinner.style.display = "none"; }, 350);
          }

          // Apply initial risk if set
          if (window.heartController && window.heartController._pendingRisk !== undefined) {
            window.heartController.setRiskLevel(window.heartController._pendingRisk);
          }
        }, function (err) {
          console.error("heart3d: Error parsing GLB:", err);
        });
      } catch (err) {
        console.error("heart3d: Failed to decode base64 GLB:", err);
      }
    }

    /* ───────────────── ANIMATION & WIGGERS CARDIAC CONTRACTION ──────── */
    let _bpm = 72;
    let _riskScore = 0.0;
    let _isPulsing = true;
    let _clock = { last: performance.now(), elapsed: 0 };

    function calculateCardiacCycle(phase) {
      let radialScale = 1.0;
      let longScale   = 1.0;
      let twistAngle  = 0.0;

      if (phase < 0.12) {
        // Isovolumetric Contraction: sharp myocardial tension onset
        const t = phase / 0.12;
        radialScale = 1.0 - Math.sin(t * Math.PI) * 0.03;
        twistAngle  = Math.sin(t * Math.PI * 0.5) * 0.04;
      } else if (phase < 0.36) {
        // Rapid Ventricular Ejection: maximal inward radial thickening & apical torsion
        const t = (phase - 0.12) / 0.24;
        const ejection = Math.sin(t * Math.PI);
        radialScale = 1.0 - ejection * 0.095; // 9.5% radial contraction
        longScale   = 1.0 - ejection * 0.065; // 6.5% base descent
        twistAngle  = 0.04 + ejection * 0.11; // ~8.6° apical torsion
      } else if (phase < 0.52) {
        // Reduced Ejection & Early Isovolumetric Relaxation: elastic recoil
        const t = (phase - 0.36) / 0.16;
        radialScale = 0.965 + t * 0.035;
        longScale   = 0.955 + t * 0.045;
        twistAngle  = (1 - t) * 0.04;
      } else if (phase < 0.85) {
        // Diastolic Filling (Diastasis): resting compliant relaxation
        radialScale = 1.0 + Math.sin((phase - 0.52) / 0.33 * Math.PI) * 0.018;
        longScale   = 1.0;
        twistAngle  = 0.0;
      } else {
        // Atrial Systole: slight pre-load ventricular stretch
        const t = (phase - 0.85) / 0.15;
        radialScale = 1.0 + Math.sin(t * Math.PI) * 0.02;
        longScale   = 1.0;
        twistAngle  = 0.0;
      }

      return { radialScale, longScale, twistAngle };
    }

    /* ───────────────── CLINICAL CONTROLLER API ─────────────────────── */
    window.heartController = {
      _pendingRisk: 0.0,

      setRiskLevel(score) {
        _riskScore = Math.max(0, Math.min(1, Number(score) || 0));
        this._pendingRisk = _riskScore;
        const risk = _riskScore;

        heartMaterials.forEach((mat) => {
          if (!mat) return;
          if (risk > 0.60) {
            // Acute / Critical Ischemic Anomaly
            const alertFactor = (risk - 0.60) / 0.40;
            // Dusky hypoperfused cyanotic hue on tissue
            mat.color.setRGB(1.0 - alertFactor * 0.18, 0.82 - alertFactor * 0.22, 0.82 - alertFactor * 0.22);
            mat.emissive.setHex(0xd9534f);
            mat.emissiveIntensity = 0.22 * alertFactor;
            rimLight.color.setHex(0xff5722);
            rimLight.intensity = 1.9;
          } else if (risk > 0.30) {
            // Moderate Risk
            const modFactor = (risk - 0.30) / 0.30;
            mat.color.setRGB(1.0, 0.94 - modFactor * 0.10, 0.94 - modFactor * 0.10);
            mat.emissive.setHex(0xeb6b34);
            mat.emissiveIntensity = 0.10 * modFactor;
            rimLight.color.setHex(0xff7582);
            rimLight.intensity = 1.5;
          } else {
            // Low Risk / Homeostasis
            mat.color.setRGB(1.0, 1.0, 1.0); // Full natural photographic texture colors
            mat.emissive.setHex(0x000000);
            mat.emissiveIntensity = 0.0;
            rimLight.color.setHex(0xff6e7a);
            rimLight.intensity = 1.4;
          }
        });
      },

      setHeartRate(bpm) {
        _bpm = Math.max(30, Math.min(200, Number(bpm) || 72));
      },

      setView(viewName) {
        controls.autoRotate = false;
        switch (viewName) {
          case "anterior": // Sternocostal view
            controls.setCameraLook(6.8, Math.PI / 2.05, 0.0);
            break;
          case "lad":      // Direct Anterior LAD & Sulcus focus
            controls.setCameraLook(5.8, Math.PI / 2.15, 0.45);
            break;
          case "lateral":  // Left Lateral Obtuse Margin (LV free wall)
            controls.setCameraLook(6.8, Math.PI / 2.0, 1.55);
            break;
          case "posterior":// Diaphragmatic Surface & Inflow Vessels
            controls.setCameraLook(6.8, Math.PI / 2.1, Math.PI);
            break;
          case "reset":
            controls.autoRotate = true;
            controls.setCameraLook(6.8, Math.PI / 2.05, 0.0);
            break;
        }
      },

      toggleAutoRotate() {
        controls.autoRotate = !controls.autoRotate;
        return controls.autoRotate;
      },

      togglePulse() {
        _isPulsing = !_isPulsing;
        return _isPulsing;
      },
    };

    /* ─────────────────── RENDER LOOP ───────────────────────────────── */
    function animate(currentTime) {
      requestAnimationFrame(animate);

      const delta = (currentTime - _clock.last) * 0.001;
      _clock.last = currentTime;
      _clock.elapsed += delta;

      const cycleDuration = 60.0 / _bpm;
      const phase = (_clock.elapsed % cycleDuration) / cycleDuration;

      if (_isPulsing) {
        const { radialScale, longScale, twistAngle } = calculateCardiacCycle(phase);
        heartGroup.scale.set(radialScale, longScale, radialScale);
        heartGroup.rotation.y = -0.28 + twistAngle;
      }

      controls.update();
      renderer.render(scene, camera);
    }

    requestAnimationFrame(animate);

    // Boot model load
    loadPhotorealisticHeart();

    // Responsive Canvas Resize
    function handleResize() {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener("resize", handleResize);
  }

  if (document.readyState === "complete" || document.readyState === "interactive") {
    initHeart();
  } else {
    document.addEventListener("DOMContentLoaded", () => initHeart());
  }
})();
