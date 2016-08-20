(function($, compatibility, profiler, jsfeat, dat) {
  "use strict";

  var stat = new profiler();

  var gui, options, ctx, canvasWidth, canvasHeight;
  var img_u8, work_canvas, work_ctx, ii_sum, ii_sqsum, ii_tilted, edg,
    ii_canny;
  var classifiers = [
        jsfeat.haar.frontalface,
        jsfeat.haar.profileface,
    ]
    // var classifier = jsfeat.haar.frontalface;

  var max_work_size = 160;

  //interest point variables
  var corners_img_u8, corners;
  //tracking variables
  var curr_img_pyr, prev_img_pyr, point_count, point_status, prev_xy, curr_xy;

  var FRAMES_BETWEEN_DETECTS = 50;
  var SHIFT_THRESH_X = 30; //max horizontal distance between faces to be considered same
  var SHIFT_THRESH_Y = 20; //max vertical distance between faces to be considered same
  var MAX_POINTS = 500;
  var MAX_TTL = 8; //max frames for a face not re-detected to be considered gone

  var lastDetectTime;
  var startTime = Date.now();
  var frames = 0;
  var score = 0;
  var faces = [],
    oldFaces = [];
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
    }
  } else {
    initVideo();
  }

  /**
   * Initialize camera video stream
   */
  function initVideo() {
    try {
      var attempts = 0;
      var readyListener = function(event) {
        findVideoSize();
      };
      var findVideoSize = function() {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          video.removeEventListener('loadeddata', readyListener);
          onDimensionsReady(video.videoWidth, video.videoHeight);
        } else {
          if (attempts < 10) {
            attempts++;
            setTimeout(findVideoSize, 200);
          } else {
            onDimensionsReady(640, 480);
          }
        }
      };
      var onDimensionsReady = function(width, height) {
        //start the app
        startApp(width, height);
        compatibility.requestAnimationFrame(tick);
      };

      video.addEventListener('loadeddata', readyListener);

      compatibility.getUserMedia({
          video: {
            facingMode: {
              exact: 'environment'
            }
          }
        },
        onGumSuccess,
        onGumError);
    } catch (error) {
      console.log(error);
      notify('Something went wrong...');
    }
  }

  /** 
   * getUserMedia callback
   */
  function onGumSuccess(stream) {
    try {
      video.src = compatibility.URL.createObjectURL(stream);
    } catch (error) {
      video.src = stream;
    }
    setTimeout(function() {
      video.play();
    }, 500);
  }

  /** 
   * getUserMedia callback
   */
  function onGumError(error) {
    notify('WebRTC not available.');
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
    this.ttl = MAX_TTL;
    this.is_stale = false;
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
  }


  var demo_opt = function() {
    //detector
    this.min_scale = 2;
    this.scale_factor = 1.15;
    this.equalize_histogram = false;
    this.use_canny = true;
    this.edges_density = 0.13;

    //points
    this.lap_thres = 10;
    this.eigen_thres = 10;

    //tracker
    this.win_size = 20;
    this.max_iterations = 30;
    this.epsilon = 0.01;
    this.min_eigen = 0.001;

    //misc
    this.detects_interval = 1000; //time in millis between detection (in that interval tracking is used on existing faces)
    this.points_per_face = 20;
    this.show_track_pts = false;
  }

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
    f1.add(options, 'edges_density', 0.01, 1.).step(0.005);
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

    var f2 = gui.addFolder('YAPE06');
    f2.add(options, "lap_thres", 1, 100);
    f2.add(options, "eigen_thres", 1, 100);
    // f2.open();

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
    f4.add(options, 'points_per_face', 5, 50).step(5);
    f4.add(options, 'show_track_pts', false);
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
    var rects = []
    for (var i = 0; i < classifiers.length; i++) {
      rects = rects.concat(jsfeat.haar.detect_multi_scale(ii_sum, ii_sqsum,
        ii_tilted, options.use_canny ? ii_canny : null, img_u8.cols,
        img_u8.rows, classifiers[i], options.scale_factor, options.min_scale
      ));
    };
    rects = jsfeat.haar.group_rectangles(rects, 1);
    if (rects.length > 0) {
      jsfeat.math.qsort(rects, 0, rects.length - 1, function(a, b) {
        return (b.confidence < a.confidence);
      })
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

  function findCorners_randCirc(faces, scale) {
    point_count = 0;

    for (var i = 0; i < faces.length; i++) {
      var face = faces[i];

      for (var j = 0; j < options.points_per_face; j++) {
        var r = Math.random() * Math.min(face.coords.w, face.coords.h) * 0.45;
        var th = Math.random() * 2 * Math.PI;

        curr_xy[point_count << 1] = ~~(r * Math.cos(th) + face.coords.cx) *
          scale;
        curr_xy[(point_count << 1) + 1] = ~~(r * Math.sin(th) + face.coords
          .cy) * scale;
        point_attr[point_count] = i;
        face.points.push(point_count);
        point_count++;
      };
    };
  }

  /**
   * ain't workin
   */
  function findCorners_yape60() {
    //assume clean frame was drawn to canvas
    var imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);

    jsfeat.imgproc.grayscale(imageData.data, canvasWidth, canvasHeight,
      corners_img_u8);

    jsfeat.imgproc.box_blur_gray(corners_img_u8, corners_img_u8, 2, 0);

    jsfeat.yape06.laplacian_threshold = options.lap_thres | 0;
    jsfeat.yape06.min_eigen_value_threshold = options.eigen_thres | 0;

    var count = jsfeat.yape06.detect(corners_img_u8, corners);

    //copy to points array for tracking
    for (point_count = 0; point_count < count; point_count++) {
      curr_xy[point_count << 1] = corners[point_count].x;
      curr_xy[(point_count << 1) + 1] = corners[point_count].y;
    };
    point_count++;

    return count;
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
    };
  }

  /**
   * Match existing faces to new detections using min-square search
   */
  function updateFaces_detect(newRects) {
    var newFaces = [];

    /**
     * calculate square distance between 2 points
     */
    function dist2(x1, y1, x2, y2) {
      return (x1-x2) * (x1-x2) + (y1 - y2) * (y1 - y2);
    }

    var indices = [];
    if (newRects.length > 0 && faces.length > 0) {
      for (var i = faces.length - 1; i >= 0; i--) {
        faces[i].is_stale = true;
        // faces[i].ttl = 0;
      };

      for (var i = 0; i < newRects.length; i++) {
        newRects[i].found = false;
      }

      var distMatrix = [];
      for (var i = 0; i < newRects.length; i++) {
        for (var j = 0; j < faces.length; j++) {
          var cx = newRects[i].x + newRects[i].width/2;
          var cy = newRects[i].y + newRects[i].height/2;
          distMatrix.push({
            r: i,
            f: j,
            d: dist2(cx, cy, faces[j].coords.cx, faces[j].coords.cy)
          });
        }
      }

      //sort
      distMatrix.sort(function(a, b) {
        return a.d - b.d;
      });
      for (var i = 0; i < distMatrix.length; i++){
        //if both face and rect are free we have a match
        if (faces[distMatrix[i].f].is_stale && !newRects[distMatrix[i].r].found){
          indices.push([distMatrix[i].r, distMatrix[i].f]);
          faces[distMatrix[i].f].is_stale = false;
          newRects[distMatrix[i].r].found = true;
        }
      }
    }

    for (var i = 0; i < indices.length; i++) {
      var rectIdx = indices[i][0];
      newRects[rectIdx].found = true;
      var rect = {
        x: newRects[rectIdx].x,
        y: newRects[rectIdx].y,
        w: newRects[rectIdx].width,
        h: newRects[rectIdx].height
      }
      rect.cx = rect.x + rect.w/2;
      rect.cy = rect.y + rect.h/2;

      var matchFace = faces[indices[i][1]];
      matchFace.setCoords(rect.x, rect.y, rect.w, rect.h);
      matchFace.is_stale = false;
      matchFace.ttl = MAX_TTL;
    }

    if (newRects.length > faces.length) {
      for (var i = 0; i < newRects.length; i++) {
        if (! newRects[i].found){
          newFaces.push(new Face(newRects[i].x, newRects[i].y, newRects[i].width, newRects[i].height));
        }
      }
    }

    //prune stale faces
    for (var i = faces.length - 1; i >= 0; i--) {
      faces[i].ttl--; 
    };
    faces = faces.filter(function(face){
      return face.ttl > 0;
    })

    //add new faces
    faces = faces.concat(newFaces);
  }

  /**
   * Match existing faces to new detections using munk-res algorithm ( https://github.com/addaleax/munkres-js )
   */
  function updateFaces_detect_(newRects) {
    var newFaces = [];

    /**
     * calculate square distance between 2 points
     */
    function dist2(x1, y1, x2, y2) {
      return (x1-x2) * (x1-x2) + (y1 - y2) * (y1 - y2);
    }

    var indices = [];
    if (newRects.length > 0 && faces.length > 0) {
      var distMatrix = [];
      for (var i = 0; i < newRects.length; i++) {
        distMatrix[i] = [];
        for (var j = 0; j < faces.length; j++) {
          var cx = newRects[i].x + newRects[i].width/2;
          var cy = newRects[i].y + newRects[i].height/2;
          distMatrix[i][j] = dist2(cx, cy, faces[j].coords.cx, faces[j].coords.cy);
        }
      }

      var m = new Munkres();
      indices = m.compute(distMatrix);
    }

    for (var i = faces.length - 1; i >= 0; i--) {
      faces[i].is_stale = true;
      // faces[i].ttl = 0;
    };

    for (var i = 0; i < newRects.length; i++) {
      newRects[i].found = false;
    }

    for (var i = 0; i < indices.length; i++) {
      var rectIdx = indices[i][0];
      newRects[rectIdx].found = true;
      var rect = {
        x: newRects[rectIdx].x,
        y: newRects[rectIdx].y,
        w: newRects[rectIdx].width,
        h: newRects[rectIdx].height
      }
      rect.cx = rect.x + rect.w/2;
      rect.cy = rect.y + rect.h/2;

      var matchFace = faces[indices[i][1]];
      matchFace.setCoords(rect.x, rect.y, rect.w, rect.h);
      matchFace.is_stale = false;
      matchFace.ttl = MAX_TTL;
    }

    if (newRects.length > faces.length) {
      for (var i = 0; i < newRects.length; i++) {
        if (! newRects[i].found){
          newFaces.push(new Face(newRects[i].x, newRects[i].y, newRects[i].width, newRects[i].height));
        }
      }
    }

    //prune stale faces
    for (var i = faces.length - 1; i >= 0; i--) {
      faces[i].ttl--; 
    };
    faces = faces.filter(function(face){
      return face.ttl > 0;
    })

    //add new faces
    faces = faces.concat(newFaces);
  }

  function updateFaces() {
    for (var i = 0; i < faces.length; i++) {
      var face = faces[i];
      var dx = 0;
      var dy = 0;

      if (face.points.length == 0) continue;

      var old_cx = 0,
        old_cy = 0,
        new_cx = 0,
        new_cy = 0;

      for (var j = 0; j < face.points.length; j++) {
        var idx = face.points[j];
        // old_cx += prev_xy[idx<<1];
        new_cx += curr_xy[idx << 1];
        // old_cy += prev_xy[(idx<<1)+1];
        new_cy += curr_xy[(idx << 1) + 1];
      };

      // old_cx /= face.points.length;
      // old_cy /= face.points.length;
      new_cx /= face.points.length;
      new_cy /= face.points.length;
      // dx = new_cx - old_cx;
      // dy = new_cy - old_cy;

      face.old_coords = face.coords;
      face.coords = Object.assign({}, face.coords, {
        x: new_cx - face.coords.w / 2,
        y: new_cy - face.coords.h / 2,
        cx: new_cx,
        cy: new_cy
      });

      // face.coords.x = new_cx - face.coords.w/2;
      // face.coords.y = new_cy - face.coords.h/2;
      // face.coords.cx = new_cx;
      // face.coords.cy = new_cy;
    };
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
        // rects = detect_haar();
        rects = detect_bbf();
        stat.stop("detector");
        lastDetectTime = Date.now();

        updateFaces_detect(rects);

        //update faces
        // oldFaces = faces;
        // faces = [];
        // for (var i = 0; i < rects.length; i++) {
        //   var newFace = new Face(rects[i].x, rects[i].y, rects[i].width, rects[i].height);
        //   faces.push(newFace);
        // };

        //find interest points
        stat.start("find points");
        // findCorners_yape60();
        findCorners_randCirc(faces, scale);
        stat.stop("find points");

        mode = 'track';
      } else if (mode == 'track') {
        // if (frames >= FRAMES_BETWEEN_DETECTS){
        if ((Date.now() - lastDetectTime) >= options.detects_interval) {
          frames = 0;
          mode = 'detect';
        }

        // do track
        stat.start("optical flow lk");
        track();
        stat.stop("optical flow lk");
        updateFaces(rects);
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

  function draw_points(ctx, points, count) {
    ctx.fillStyle = "rgb(0, 255, 0)";
    for (var i = 0; i < count; i++) {
      var x = points[i << 1];
      var y = points[(i << 1) + 1];
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2, true);
      ctx.closePath();
      ctx.fill();
    };
  }

  function draw_faces_detect(ctx, rects, sc, max) {
    ctx.strokeStyle = "rgb(0, 255, 0)";
    var on = rects.length;
    // if(on && max) {
    //     jsfeat.math.qsort(rects, 0, on-1, function(a,b){return (b.confidence<a.confidence);})
    // }
    var n = max || on;
    n = Math.min(n, on);
    var r;
    for (var i = 0; i < n; ++i) {
      r = rects[i];
      var size = 6;
      var x = (r.x * sc + r.width * sc * 0.5 - size * 0.5) | 0;
      var y = (r.y * sc + r.height * sc * 0.5 - size * 0.5) | 0;
      ctx.fillRect(x, y, size, size);
      ctx.strokeRect((r.x * sc) | 0, (r.y * sc) | 0, (r.width * sc) | 0, (r
        .height * sc) | 0);
    }
  }


  /**
   *
   * @param ctx canvas context to draw on
   * @param faces array of Face objects to draw
   * @param sc scale factor from working canvas to output canvas
   * @max   max number of rectangles to draw
   */
  function draw_faces_track(ctx, faces, sc, max) {
    ctx.fillStyle = "rgb(0,255,128)";
    ctx.strokeStyle = "rgb(0,255,128)";


    var on = faces.length;
    if (on && max) {
      //todo: sort by confidence
    }
    var n = max || on;
    n = Math.min(n, on);
    var face;
    for (var i = 0; i < n; ++i) {
      face = faces[i];

      // Rescale coordinates from detector to video coordinate space:
      // var x = face.cx * video.videoWidth / canvasWidth;
      // var y = face.cy * video.videoHeight / canvasHeight;

      var size = 30;

      var cx = face.coords.cx * sc;
      var cy = face.coords.cy * sc;
      var rad = Math.min(face.coords.w, face.coords.h) / 2;

      ctx.beginPath();
      ctx.arc(face.coords.cx, face.coords.cy, rad, 0, 2 * Math.PI, true);
      ctx.stroke();
      // ctx.strokeRect(face.coords.x, face.coords.y, face.coords.w, face.coords.h);

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