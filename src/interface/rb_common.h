
#ifndef __RB_COMMON_H__
#define __RB_COMMON_H__

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef NULL
#define NULL 0L
#endif
#define RB_NULL     0L

#define RB_OPMAX(a, b) ((a) > (b) ? (a) : (b))
#define RB_OPMIN(a, b) ((a) < (b) ? (a) : (b))

/** RB_EXPORTS */
#if defined(RB_API_EXPORTS)
#define RB_EXPORTS __declspec(dllexport)
#elif defined(RB_API_IMPORTS)
#define RB_EXPORTS __declspec(dllimport)
#else
#define RB_EXPORTS extern
#endif

/** RB_INLINE */
#ifndef RB_INLINE
#if defined __cplusplus
#define RB_INLINE inline
#elif (defined WIN32 || defined WIN64) && !defined __GNUC__
#define RB_INLINE __inline
#else
#define RB_INLINE static
#endif
#endif

    /* META type definitions */
    typedef unsigned char   RB_U8;
    typedef unsigned short  RB_U16;
    typedef unsigned int    RB_U32;
    
    typedef char            RB_S8;
    typedef short           RB_S16;
    typedef int             RB_S32;

#ifndef _M_IX86
    typedef unsigned long long RB_U64;
    typedef long long RB_S64;
#else
    typedef __int64 RB_U64;
    typedef __int64 RB_S64;
#endif

    typedef float           RB_FLOAT;
    typedef double          RB_DOUBLE;
    typedef void            RB_VOID;
    typedef unsigned long   RB_SIZE_T;

    /* Handle */
    typedef void *          RB_HANDLE, RB_VOID_P;

    /* Boolean */
    typedef enum _rb_bool_e
    {
        RB_FALSE = 0,
        RB_TRUE  = 1,
    } RB_BOOL;

    /* shape enum */
    typedef enum _rb_shape_e
    {
        RB_SHAPE_POINT    = 0,
        RB_SHAPE_LINE     = 1,
        RB_SHAPE_RECT     = 2,
        RB_SHAPE_CIRCLE   = 3,
        RB_SHAPE_ELLIPSE  = 4,
        RB_SHAPE_POLYGON  = 5,
        RB_SHAPE_POLYLINE = 6,
        RB_SHAPE_FREEHAND = 7
    } RB_SHAPE_E;

    /* Point Definition */
    typedef struct _rb_2dpoint_s
    {
        RB_S32 s32X;    /* X */
        RB_S32 s32Y;    /* Y */
    } RB_POINT_S, RB_2DPOINT_S;

    /* 3D point */
    typedef struct _rb_3dpoint_s
    {
        RB_S32 s32X;    /* X */
        RB_S32 s32Y;    /* Y */
        RB_S32 s32Z;    /* Z */
    } RB_3DPOINT_S;

    /* 2d point with float data */
    typedef struct _rb_2dpointf_s
    {
        RB_FLOAT f32X;    /* X */
        RB_FLOAT f32Y;    /* Y */
    } RB_POINTF_S, RB_2DPOINTF_S;

    /* 3d point with float data */
    typedef struct _rb_3dpointf_s
    {
        RB_FLOAT f32X;    /* X */
        RB_FLOAT f32Y;    /* Y */
        RB_FLOAT f32Z;    /* Z */
    } RB_3DPOINTF_S;

    /* line */
    typedef struct _rb_line_s  
    {
        RB_POINT_S stStart; /* Start Point */
        RB_POINT_S stEnd;   /* End Point */
    } RB_LINE_S;

    /* line with float data */
    typedef struct _rb_linef_s
    {
        RB_POINTF_S stFStart; /* Start Point */
        RB_POINTF_S stFEnd;   /* End Point */
    } RB_LINEF_S;

    /* rect */
    typedef struct _rb_rect_s 
    {
        RB_POINT_S stTopLeft;     /* Top Left Point */
        RB_POINT_S stBottomRight; /* Bottom Right Point */
    } RB_RECT_S;

    /* rect with float data */
    typedef struct _rb_rectf_s
    {
        RB_POINTF_S stFTopLeft;     /* Top Left Point */
        RB_POINTF_S stFBottomRight; /* Bottom Right Point */
    } RB_RECTF_S;

    /* rotate rect */
    typedef struct _rb_rotate_rect_s
    {
        RB_RECT_S stRect;       /* Rectangle */
        RB_FLOAT  f32Angle;     /* Angle */
    } RB_ROTATE_RECT_S;

    /* circle */
    typedef struct _rb_circle_s
    {
        RB_POINT_S stCenter;     /* Center Point */
        RB_S32     s32Radius;    /* Radius */
    } RB_CIRCLE_S;

    /* float circle */
    typedef struct _rb_circlef_s
    {
        RB_POINTF_S stCenter;     /* Center Point */
        RB_FLOAT    f32Radius;    /* Radius */
    } RB_CIRCLEF_S;

    /* ellipse */
    typedef struct _rb_ellipse_s
    {
        RB_POINT_S stCenter;     /* Center Point */
        RB_S32     s32RadiusX;   /* Radius X */
        RB_S32     s32RadiusY;   /* Radius Y */
    } RB_ELLIPSE_S;

    /* float ellipse */
    typedef struct _rb_ellipsef_s
    {
        RB_POINTF_S stCenter;     /* Center Point */
        RB_FLOAT    f32RadiusX;   /* Radius X */
        RB_FLOAT    f32RadiusY;   /* Radius Y */
    } RB_ELLIPSEF_S;

    /**
     * @brief Multi-points definition.
     * 
     * RB_POLYGON_S: adjacent point order loop, first->two->three....->last->first, 
     *      closed plane area, polygon shape.
     * RB_POLYLINE_S: adjacent point order, first->two->three....->last, open no loop
     * RB_POINTS_SET_S: adjacent point order, first, two, three, ...., last, points set
     * RB_FREEHAND_S: just points set, no order, no loop
     */
    typedef struct _rb_multipts_s
    {
        RB_S32 s32Num;         /* Number of Points */
        RB_POINT_S* pstPoints; /* pt sets */
    } RB_POLYGON_S, RB_POLYLINE_S, RB_POINTS_SET_S, RB_FREEHAND_S;

    /**
     * @brief Multi-lines definition.
     * 
     * maybe used for key-points set, or line-sets, or curve-sets, or ...
     * bIsDirect: Defaults to RB_FALSE.
     *     if true, the line is direct from start to end, otherwise, the line is not direct
     */
    typedef struct _rb_multlines_s
    {
        RB_S32     s32Num;        /* Number of Lines */
        RB_BOOL    bIsDirect;     /* Direct or not */
        RB_LINE_S* pstLines;      /* lines */
    } RB_LINES_S;

    /**
     * @brief Deep learning or Machine Learning object.
     */
    typedef struct _rb_dlmltgt_s
    {
        RB_S32     s32LabelID;    /* Label ID */
        RB_RECT_S  stRect;        /* Rectangle */
        RB_FLOAT   f32Prob;       /* Probability */
    } RB_Target_S;
    
    /**
     * @brief if f32Real = 2, f32Imag = 3, then the complex number is 2+3i
     */     
    typedef struct _rb_complexdata_s
    {
        RB_FLOAT f32Real;
        RB_FLOAT f32Imag;
    }RB_Complex_S;
    

    /**
     * @brief all zero is ok, good status
     * we can use the 'binary shift' and 'binary xor' or 'binary or' to get the status
     */ 
    typedef enum _rb_status_code_e
    {
        RB_SUCCESS                 = 0x00000000,    /* Success */
        RB_FAILURE                 = 0x80000000,    /* Failure */
    
        /* memory failure */     
        RB_MEM_FAIL                = 0x80000001,    /* memory failure */
        RB_MEM_FAIL_MALLOC         = 0x80000002,    /* memory allocation failure */
        RB_MEM_FAIL_CALLOC         = 0x80000003,    /* calloc */
        RB_MEM_FAIL_REALLOC        = 0x80000004,    /* realloc failure */
        RB_MEM_FAIL_FREE           = 0x80000005,    /* free memory failure */

        /* image failure */ 
        RB_IMAGE_FAIL              = 0x80000010,    /* image failure */
        RB_IMAGE_FAIL_SIZE         = 0x80000020,    /* image size error */
        RB_IMAGE_FAIL_FORMAT       = 0x80000030,    /* image format error */
        RB_IMAGE_FAIL_READ         = 0x80000040,    /* read image file failure */
        RB_IMAGE_FAIL_WRITE        = 0x80000050,    /* write failure */
    
        /* file failure */     
        RB_FILE_FAIL               = 0x80000100,    /* file operation failure */
        RB_FILE_FAIL_READ          = 0x80000200,    /* read file failure */
        RB_FILE_FAIL_WRITE         = 0x80000300,    /* write file failure */
    
        /* para failure */     
        RB_PARA_FAIL               = 0x80001000,    /* parameter error */
        RB_PARA_FAIL_CONFIG        = 0x80002000,    /* config file failure */
        RB_PARA_FAIL_INVALID       = 0x80003000,    /* invalid parameter */
        RB_PARA_FAIL_UNSUPPORT     = 0x80003000,    /* not support */
    
        /* license failure */    
        RB_LICENSE_FAIL            = 0x80010000,   /* license failure */
        RB_LICENSE_FAIL_TIMEOUT    = 0x80020000,   /* license timeout */
        RB_LICENSE_FAIL_READ_MAC   = 0x80030000,   /* read mac failure */
        RB_LICENSE_FAIL_CHECK      = 0x80040000,   /* license check failure */
        RB_LICENSE_FAIL_INVALID    = 0x80050000,   /* license invalid */

        /* net failure */
        RB_NETWORK_FAIL            = 0x80100000,   /* network failure */
        RB_NETWORK_CONNECT         = 0x80200000,   /* network connect failure */
        RB_NETWORK_PUSH            = 0x80300000,   /* network push failure */
        RB_NETWORK_GET             = 0x80400000,   /* network get failure */
    } RB_STATUS_CODE_E;

    /**
     * @brief the packed pixel, maybe rgb, yuv, hsv, gray, ...
     */
    typedef struct _rb_packed_pixel_s
    {
        RB_U8 u8ColorA;
        RB_U8 u8ColorB;
        RB_U8 u8ColorC;
        RB_U8 u8ColorReserved;
    } RB_PACKED_PIXEL_S;
    
    /**
     * @brief usual image format
     *  U(Cb), V(Cr)
     */
    typedef enum _rb_image_format_e
    {
        RB_IMAGE_FORMAT_UNKNOWN = 0,
        RB_IMAGE_FORMAT_GRAY,            /* GRAY GRAY GRAY GRAY */
        RB_IMAGE_FORMAT_RGB_PACKED,      /* RGB RGB RGB RGB RGB RGB RGB RGB */
        RB_IMAGE_FORMAT_RGB_PLANAR,      /* RRRRRRRR GGGGGGGG BBBBBBBBB */ 
        RB_IMAGE_FORMAT_BGR_PACKED,      /* BGR BGR BGR BGR BGR BGR BGR BGR */ 
        RB_IMAGE_FORMAT_BGR_PLANAR,      /* BBBBBBBBB GGGGGGGG RRRRRRRR */ 
        RB_IMAGE_FORMAT_HSV,             /* hsv, but not often used */
        RB_IMAGE_FORMAT_HLS,             /* hls, but not often used */
        RB_IMAGE_FORMAT_YUV444P,         /* YYYYYYYY VVVVVVVV UUUUUUU */
        RB_IMAGE_FORMAT_YUV422P,         /* YYYYYYYY VVVV UUUU */
        RB_IMAGE_FORMAT_YUV422_YUYV,     /* YUYV YUYV YUYV YUYV */
        RB_IMAGE_FORMAT_YUV422_UYVY,     /* UYVY UYVY UYVY UYVY */
        RB_IMAGE_FORMAT_YUV420p_YV12,    /* YYYYYYYY VV UU */
        RB_IMAGE_FORMAT_YUV420p_I420,    /* YYYYYYYY UU VV */
        RB_IMAGE_FORMAT_YUV420sp,        /* YYYYYYYY UVUV, default */
        RB_IMAGE_FORMAT_YUV420_NV12,     /* YYYYYYYY UVUV */
        RB_IMAGE_FORMAT_YUV420_NV21,     /* YYYYYYYY VUVU */
        RB_IMAGE_FORMAT_YUV400,          /* YYYYYYYY, only y */
        RB_IMAGE_FORMAT_BayerRGGB,       /* RGGB RGGB RGGB RGGB RGGB RGGB */
        RB_IMAGE_FORMAT_BayerGRBG,       /* GRBG GRBG GRBG GRBG GRBG GRBG */
        RB_IMAGE_FORMAT_BayerGBRG,       /* GBRG GBRG GBRG GBRG GBRG GBRG */
        RB_IMAGE_FORMAT_BayerBGGR,       /* BGGR BGGR BGGR BGGR BGGR BGGR */
        RB_IMAGE_FORMAT_BayerGR,         /* GBRG GBRG GBRG GBRG GBRG GBRG */
        RB_IMAGE_FORMAT_BayerRG,         /* RGGB RGGB RGGB RGGB RGGB RGGB */
        RB_IMAGE_FORMAT_BayerGB,         /* GBRG GBRG GBRG GBRG GBRG GBRG */
        RB_IMAGE_FORMAT_BayerBG,         /* BGGR BGGR BGGR BGGR BGGR BGGR */
        RB_IMAGE_FORMAT_UNSUPPORTED,     /* unsupported format */
    }RB_IMAGE_FORMAT_E;

    /**
     * @brief
     * do remember that the image is a matrix, not a vector
     * do remember to free the memory after use
     * do remember how to store the data in the memory accord to the image format
     *
     */
        typedef struct _rb_usual_image_s
    {
        RB_S32            s32W;          /* width */
        RB_S32            s32H;          /* height */
        RB_U8*            pData;         /* image data */
        RB_IMAGE_FORMAT_E eFormat;       /* format */
    } RB_IMAGE_S;


    /**
     * @brief 
     * do remember how to store the data in the memory.
     *       if the mat store the image, do remember the data organization
     * do remember to free the memory after use.
     */
    typedef struct _rb_mat_s
    {
        RB_S32   s32C;    /* Channel */
        RB_S32   s32W;    /* Width */
        RB_S32   s32H;    /* Height */
        RB_VOID* pData;   /* Data */
    } RB_MAT_S, RB_EXIMAGE_S;

#ifdef __cplusplus
}
#endif

#endif /*_RB_COMMON_H_*/
