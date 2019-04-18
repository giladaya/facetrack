(function($, compatibility, profiler, jsfeat, dat) {
  "use strict";

  var stat = new profiler();

  var gui, options, ctx, canvasWidth, canvasHeight;
  var img_u8, work_canvas, work_ctx, ii_sum, ii_sqsum, ii_tilted, edg, ii_canny;
  var classifiers = [
        jsfeat.haar.frontalface,
        // jsfeat.haar.profileface,
    ];
    // var classifier = jsfeat.haar.frontalface;

  var max_work_size = 160;

  //interest point variables
  var corners_img_u8, corners;
  //tracking variables
  var curr_img_pyr, prev_img_pyr, point_count, point_status, prev_xy, curr_xy;

  var FRAMES_BETWEEN_DETECTS = 50;
  var SHIFT_THRESH2 = 10000; //max horizontal distance between faces to be considered same
  var SHIFT_THRESH_X = 30; //max horizontal distance between faces to be considered same
  var SHIFT_THRESH_Y = 20; //max vertical distance between faces to be considered same
  var MAX_POINTS = 500;
  var MIN_TTL = 1; //min detect cycles for a face not re-detected to be considered gone
  var MAX_TTL = 10; //max detect cycles for a face not re-detected to be considered gone

  var lastDetectTime;
  var startTime = Date.now();
  var frames = 0;
  var faces = [];
  var point_attr; //point to face lookup

  // lets do some fun
  var video = document.getElementById('webcam');
  var canvas = document.getElementById('canvas');

  //make sure we init with the correct orientation
  if ('orientation' in screen && screen.orientation.type.startsWith('portrait')) {
    screen.orientation.onchange = function() {
      if (screen.orientation.type.startsWith('landscape')) {
        initVideo();
        screen.orientation.onchange = null;
      }
    };
  } else {
    initVideo();
  }

  /**
   * Initialize camera video stream
   */
  function initVideo() {
    // Prefer camera resolution nearest to 1280x720.
    // var constraints = { audio: true, video: { width: 1280, height: 720 } }; 
    var constraints = { video: { facingMode: "user" } };

    navigator.mediaDevices.getUserMedia(constraints)
    .then(function(mediaStream) {
      video.srcObject = mediaStream;
      video.onloadedmetadata = function(e) {
        console.log(e);
        const video = e.srcElement;

        video.play();
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        startApp(width, height);
        compatibility.requestAnimationFrame(tick);
      };
    })
    .catch(function(err) { // always check for errors at the end.
      notify('WebRTC not available.');
      console.log(err.name + ": " + err.message); 
    }); 
  }

  /** 
   * Notify of an error
   */
  function notify(msg) {
    $('#err').html(msg);
    $('#err').show();
  }

  /** 
   * Constructor for a Face object
   * @param cx face center x
   * @param cy face center y
   */
  function Face(x, y, w, h) {
    var coords = {
      'x': x,
      'y': y,
      'w': w,
      'h': h,
      'cx': x + w / 2,
      'cy': y + h / 2
    };

    this.coords = coords;
    this.old_coords = coords;
    this.id = Math.ceil(Math.random() * 10000);
    this.ttl = MIN_TTL;
    this.age = 0; //age in detection cycles
    this.is_match = false;
    this.is_live = true;
    this.points = [];
  }

  Face.prototype.setCoords = function(x, y, w, h) {
    w = w || this.coords.w;
    h = h || this.coords.h;
    this.old_coords = this.coords;
    var cx = x + w / 2;
    var cy = y + h / 2;
    this.coords = Object.assign({}, this.coords, {
      x: x,
      y: y,
      w: w, 
      h: h,
      cx: cx,
      cy: cy
    });
  };

  var demo_opt = function() {
    //detector
    this.min_scale = 2;
    this.scale_factor = 1.15;
    this.equalize_histogram = false;
    this.use_canny = true;
    this.edges_density = 0.13;

    //points
    this.points_per_face = 20;
    this.show_track_pts = false;

    //tracker
    this.win_size = 20;
    this.max_iterations = 30;
    this.epsilon = 0.01;
    this.min_eigen = 0.001;

    //misc
    this.detector = 'BBF';
    this.detects_interval = 1000; //time in millis between detection (in that interval tracking is used on existing faces)
  };

  function startApp(videoWidth, videoHeight) {
    if (window.innerHeight > window.innerWidth) {

    }
    canvasWidth = canvas.width;
    // canvasHeight = canvas.height;
    canvasHeight = ~~(canvas.width * window.innerHeight / window.innerWidth);
    canvas.height = canvasHeight;
    ctx = canvas.getContext('2d');

    ctx.fillStyle = "rgb(0,255,0)";
    ctx.strokeStyle = "rgb(0,255,0)";

    var scale = Math.min(max_work_size / videoWidth, max_work_size /
      videoHeight);
    var workWidth = (videoWidth * scale) | 0;
    var workHeight = (videoHeight * scale) | 0;

    img_u8 = new jsfeat.matrix_t(workWidth, workHeight, jsfeat.U8_t |
      jsfeat.C1_t);
    edg = new jsfeat.matrix_t(workWidth, workHeight, jsfeat.U8_t | jsfeat.C1_t);
    work_canvas = document.createElement('canvas');
    work_canvas.width = workWidth;
    work_canvas.height = workHeight;
    work_ctx = work_canvas.getContext('2d');
    ii_sum = new Int32Array((workWidth + 1) * (workHeight + 1));
    ii_sqsum = new Int32Array((workWidth + 1) * (workHeight + 1));
    ii_tilted = new Int32Array((workWidth + 1) * (workHeight + 1));
    ii_canny = new Int32Array((workWidth + 1) * (workHeight + 1));

    options = new demo_opt();
    gui = new dat.GUI();

    var f1 = gui.addFolder('HAAR');
    f1.add(options, 'min_scale', 1, 4).step(0.1);
    f1.add(options, 'scale_factor', 1.1, 2).step(0.025);
    f1.add(options, 'equalize_histogram');
    f1.add(options, 'use_canny');
    f1.add(options, 'edges_density', 0.01, 1.0).step(0.005);
    // f1.open();

    stat.add("detector");

    //init interest points
    corners_img_u8 = new jsfeat.matrix_t(canvasWidth, canvasHeight, jsfeat.U8_t |
      jsfeat.C1_t);
    corners = [];
    var i = canvasWidth * canvasHeight;
    while (--i >= 0) {
      corners[i] = new jsfeat.keypoint_t(0, 0, 0, 0);
    }

    var f2 = gui.addFolder('POINTS');
    f2.add(options, 'points_per_face', 5, 50).step(5);
    f2.add(options, 'show_track_pts', false);
    f2.open();

    stat.add("find points");

    //init LK tracker
    curr_img_pyr = new jsfeat.pyramid_t(3);
    prev_img_pyr = new jsfeat.pyramid_t(3);
    curr_img_pyr.allocate(canvasWidth, canvasHeight, jsfeat.U8_t | jsfeat.C1_t);
    prev_img_pyr.allocate(canvasWidth, canvasHeight, jsfeat.U8_t | jsfeat.C1_t);

    point_count = 0;
    point_status = new Uint8Array(MAX_POINTS);
    point_attr = new Uint8Array(MAX_POINTS);
    prev_xy = new Float32Array(MAX_POINTS * 2);
    curr_xy = new Float32Array(MAX_POINTS * 2);

    var f3 = gui.addFolder('KLT');
    f3.add(options, 'win_size', 7, 30).step(1);
    f3.add(options, 'max_iterations', 3, 30).step(1);
    f3.add(options, 'epsilon', 0.001, 0.1).step(0.0025);
    f3.add(options, 'min_eigen', 0.001, 0.01).step(0.0025);
    // f3.open();

    stat.add("optical flow lk");

    var f4 = gui.addFolder('misc');
    f4.add(options, 'detects_interval', 0, 10000).step(100);
    f4.add(options, 'detector', ['BBF', 'HAAR']);
    f4.open();

    //init bbf detector
    jsfeat.bbf.prepare_cascade(jsfeat.bbf.face_cascade);

    gui.close();
  }

  /**
   * Detect objects using HAAR algorithm
   */
  function detect_haar() {
    work_ctx.drawImage(video, 0, 0, work_canvas.width, work_canvas.height);
    var imageData = work_ctx.getImageData(0, 0, work_canvas.width,
      work_canvas.height);

    jsfeat.imgproc.grayscale(imageData.data, work_canvas.width, work_canvas
      .height, img_u8);

    // possible options
    if (options.equalize_histogram) {
      jsfeat.imgproc.equalize_histogram(img_u8, img_u8);
    }
    jsfeat.imgproc.gaussian_blur(img_u8, img_u8, 3);

    jsfeat.imgproc.compute_integral_image(img_u8, ii_sum, ii_sqsum, (
      'tilted' in classifiers[0]) ? ii_tilted : null);

    if (options.use_canny) {
      jsfeat.imgproc.canny(img_u8, edg, 10, 50);
      jsfeat.imgproc.compute_integral_image(edg, ii_canny, null, null);
    }

    jsfeat.haar.edges_density = options.edges_density;
    var rects = [];
    for (var i = 0; i < classifiers.length; i++) {
      rects = rects.concat(jsfeat.haar.detect_multi_scale(ii_sum, ii_sqsum,
        ii_tilted, options.use_canny ? ii_canny : null, img_u8.cols,
        img_u8.rows, classifiers[i], options.scale_factor, options.min_scale
      ));
    }
    rects = jsfeat.haar.group_rectangles(rects, 1);
    if (rects.length > 0) {
      jsfeat.math.qsort(rects, 0, rects.length - 1, function(a, b) {
        return (b.confidence < a.confidence);
      });
    }

    //scale all points
    var scale = canvasWidth / img_u8.cols;
    rects = rects.map(function(r) {
      var nr = Object.assign({}, r);
      nr.x *= scale;
      nr.y *= scale;
      nr.width *= scale;
      nr.height *= scale;
      return nr;
    });

    return rects;
  }

  /**
   * Detect faces using BBF algorithm
   */
  function detect_bbf() {
    work_ctx.drawImage(video, 0, 0, work_canvas.width, work_canvas.height);
    var imageData = work_ctx.getImageData(0, 0, work_canvas.width,
      work_canvas.height);

    jsfeat.imgproc.grayscale(imageData.data, work_canvas.width, work_canvas
      .height, img_u8);

    // possible options
    //jsfeat.imgproc.equalize_histogram(img_u8, img_u8);

    var pyr = jsfeat.bbf.build_pyramid(img_u8, 24 * 2, 24 * 2, 4);

    var rects = jsfeat.bbf.detect(pyr, jsfeat.bbf.face_cascade);
    rects = jsfeat.bbf.group_rectangles(rects, 1);

    //scale all points
    var scale = canvasWidth / img_u8.cols;
    rects = rects.map(function(r) {
      var nr = Object.assign({}, r);
      nr.x *= scale;
      nr.y *= scale;
      nr.width *= scale;
      nr.height *= scale;
      return nr;
    });

    return rects;
  }

  /**
   * Randomly pick points to track around the center of each face
   */
  function findCorners_randCirc(faces, scale) {
    point_count = 0;

    for (var i = 0; i < faces.length; i++) {
      var face = faces[i];

      for (var j = 0; j < options.points_per_face; j++) {
        var r = Math.random() * Math.min(face.coords.w, face.coords.h) * 0.35;
        var th = Math.random() * 2 * Math.PI;

        curr_xy[point_count << 1] = ~~(r * Math.cos(th) + face.coords.cx) *
          scale;
        curr_xy[(point_count << 1) + 1] = ~~(r * Math.sin(th) + face.coords
          .cy) * scale;
        point_attr[point_count] = i;
        face.points.push(point_count);
        point_count++;
      }
    }
  }

 
  /**
   * track all tracking points using LK optical flow algorithm
   */
  function track() {
    var imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);

    // swap flow data
    var _pt_xy = prev_xy;
    prev_xy = curr_xy;
    curr_xy = _pt_xy;
    var _pyr = prev_img_pyr;
    prev_img_pyr = curr_img_pyr;
    curr_img_pyr = _pyr;

    jsfeat.imgproc.grayscale(imageData.data, canvasWidth, canvasHeight,
      curr_img_pyr.data[0]);

    curr_img_pyr.build(curr_img_pyr.data[0], true);

    jsfeat.optical_flow_lk.track(prev_img_pyr, curr_img_pyr, prev_xy,
      curr_xy, point_count, options.win_size | 0, options.max_iterations |
      0, point_status, options.epsilon, options.min_eigen);

    //prune_oflow_points(ctx);
    //filter points in faces
    for (var i = 0; i < faces.length; i++) {
      faces[i].points = [];
    }

    var n = point_count;
    var i = 0,
      j = 0;

    for (; i < n; ++i) {
      if (point_status[i] == 1) {
        if (j < i) {
          curr_xy[j << 1] = curr_xy[i << 1];
          curr_xy[(j << 1) + 1] = curr_xy[(i << 1) + 1];
          point_attr[j] = point_attr[i];
        }
        ++j;
      }
    }
    point_count = j;

    for (i = 0; i < point_count; i++) {
      faces[point_attr[i]].points.push(i);
    }
  }

  /**
   * Match existing faces to new detections using min-distance search
   */
  function updateFaces_detect(newRects) {
    var newFaces = [];

    /**
     * calculate square distance between 2 points
     */
    function dist2(x1, y1, x2, y2) {
      return (x1-x2) * (x1-x2) + (y1 - y2) * (y1 - y2);
    }

    //update age and ttl for all existing faces
    faces.forEach(function(face, idx, array) {
      face.ttl--;
      face.age++;
    });

    var indices = [];
    if (newRects.length > 0 && faces.length > 0) {
      //mark all faces and all rects as unmatched
      faces.forEach(function(face, idx, array) {
        face.is_match = false;
      });
      newRects.forEach(function(rect, idx, array) {
        rect.is_match = false;
      });

      //calculate distances matrix
      var distMatrix = [];
      // for (var i = 0; i < newRects.length; i++) {
      //   for (var j = 0; j < faces.length; j++) {
      //     var cx = newRects[i].x + newRects[i].width/2;
      //     var cy = newRects[i].y + newRects[i].height/2;
      //     distMatrix.push({
      //       r: i,
      //       f: j,
      //       d: dist2(cx, cy, faces[j].coords.cx, faces[j].coords.cy)
      //     });
      //   }
      // }
      newRects.forEach (function(rect, i){
        faces.forEach(function(face, j){
          var cx = rect.x + rect.width/2;
          var cy = rect.y + rect.height/2;
          distMatrix.push({
            r: i,
            f: j,
            d: dist2(cx, cy, face.coords.cx, face.coords.cy)
          });
        });
      });


      //sort distances
      distMatrix.sort(function(a, b) {
        return a.d - b.d;
      });

      //find matches
      for (var i = 0; i < distMatrix.length; i++){
        if (distMatrix[i].d > SHIFT_THRESH2) {
          //distances from here on are larger than the threshold
          break;
        }
        //if both face and rect are unmatched we have a match
        if (!faces[distMatrix[i].f].is_match && !newRects[distMatrix[i].r].is_match){
          indices.push([distMatrix[i].r, distMatrix[i].f]);
          faces[distMatrix[i].f].is_match = true;
          newRects[distMatrix[i].r].is_match = true;
        }
      }
    }

    //update matched faces with new bounding box
    indices.forEach(function(el, idx, array) {
      var rectIdx = el[0];
      newRects[rectIdx].is_match = true;
      var rect = {
        x: newRects[rectIdx].x,
        y: newRects[rectIdx].y,
        w: newRects[rectIdx].width,
        h: newRects[rectIdx].height
      };
      rect.cx = rect.x + rect.w/2;
      rect.cy = rect.y + rect.h/2;

      var matchFace = faces[el[1]];
      matchFace.setCoords(rect.x, rect.y, rect.w, rect.h);
      matchFace.ttl = Math.max(Math.min(matchFace.age, MAX_TTL), MIN_TTL);
    });

    //add unmatched faces
    // if (newRects.length > faces.length) {
      newRects.forEach(function(rect, idx, array) {
        if (! rect.is_match){
          newFaces.push(new Face(rect.x, rect.y, rect.width, rect.height));
        }
      });
    // }

    //prune stale faces
    faces = faces.filter(function(face){
      return face.ttl > 0;
    });


    //add new faces
    faces = faces.concat(newFaces);
  }

  /**
   * Update location of tracked faces according to optical flow results
   */
  function updateFaces_track() {
    for (var i = 0; i < faces.length; i++) {
      var face = faces[i];
      if (face.points.length === 0) continue;

      var new_cx = 0,
          new_cy = 0;

      for (var j = 0; j < face.points.length; j++) {
        var idx = face.points[j];
        new_cx += curr_xy[idx << 1];
        new_cy += curr_xy[(idx << 1) + 1];
      }

      new_cx /= face.points.length;
      new_cy /= face.points.length;

      face.setCoords(new_cx - face.coords.w / 2, new_cy - face.coords.h / 2);
    }
  }

  var mode = 'detect';
  var rects = [];

  function tick() {
    compatibility.requestAnimationFrame(tick);
    stat.new_frame();
    if (video.readyState === video.HAVE_ENOUGH_DATA) {

      // ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
      ctx.drawImage(video, 0, 0);
      var scale = 1;

      if (mode == 'detect') {
        stat.start("detector");
        if (options.detector == 'HAAR') {
          rects = detect_haar();
        } else {
          rects = detect_bbf();
        }
        stat.stop("detector");
        lastDetectTime = Date.now();

        updateFaces_detect(rects);

        //find interest points
        stat.start("find points");
        findCorners_randCirc(faces, scale);
        stat.stop("find points");

        mode = 'track';
      } else if (mode == 'track') {
        if ((Date.now() - lastDetectTime) >= options.detects_interval) {
          frames = 0;
          mode = 'detect';
        }

        // do track
        stat.start("optical flow lk");
        track();
        stat.stop("optical flow lk");
        updateFaces_track();
      } else {
        //do nothing
      }


      draw_faces_detect(ctx, rects, scale, 5);
      draw_faces_track(ctx, faces, scale, 5);
      if (options.show_track_pts) {
        draw_points(ctx, curr_xy, point_count);
      }

      frames++;
    }
    $('#log').html(stat.log());
  }

  /**
   * Draw all tracking points
   */
  function draw_points(ctx, points, count) {
    ctx.fillStyle = "rgb(0, 255, 0)";
    for (var i = 0; i < count; i++) {
      var x = points[i << 1];
      var y = points[(i << 1) + 1];
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2, true);
      ctx.closePath();
      ctx.fill();
    }
  }

  /**
   * Draw location and size of deteted faces - this updates each interval
   */
  function draw_faces_detect(ctx, rects, sc, max) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgb(0, 255, 0)";
    var on = rects.length;
    var n = max || on;
    n = Math.min(n, on);
    var r;
    for (var i = 0; i < n; ++i) {
      r = rects[i];
      ctx.strokeRect((r.x * sc) | 0, (r.y * sc) | 0, (r.width * sc) | 0, (r
        .height * sc) | 0);
    }
  }


  /**
   * Draw location and size of tracked faces
   * @param ctx canvas context to draw on
   * @param faces array of Face objects to draw
   * @param sc scale factor from working canvas to output canvas
   * @max   max number of rectangles to draw
   */
  function draw_faces_track(ctx, faces, sc, max) {
    ctx.fillStyle = "rgb(0,255,128)";
    ctx.strokeStyle = "rgb(0,255,128)";
    ctx.lineWidth = 3;


    var on = faces.length;
    if (on && max) {
      //todo: sort by confidence
    }
    var n = max || on;
    n = Math.min(n, on);
    var face;
    for (var i = 0; i < n; ++i) {
      face = faces[i];
      if (face.age < 1) {
        //continue;
      }

      var rad = Math.min(face.coords.w, face.coords.h) / 2;

      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(face.coords.cx, face.coords.cy, rad, 0, 2 * Math.PI, false);
      ctx.stroke();

      ctx.lineWidth = 4;      
      ctx.beginPath();
      ctx.arc(face.coords.cx, face.coords.cy, rad, 0, (face.ttl / (MAX_TTL-1)) * 2 * Math.PI, false);
      ctx.stroke();

      ctx.font = "24px Verdana";
      ctx.fillStyle = "rgb(255,255,255)";
      ctx.fillText(face.id, face.coords.x, face.coords.y);
    }
  }


  $(window).unload(function() {
    video.pause();
    video.src = null;
  });
})($, compatibility, profiler, jsfeat, dat);