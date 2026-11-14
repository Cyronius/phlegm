module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(0, 2)
    %logical_core_0 = aie.logical_tile<CoreTile>(0, 3)
    %logical_core_1 = aie.logical_tile<CoreTile>(0, 4)
    %logical_core_2 = aie.logical_tile<CoreTile>(0, 5)
    %logical_core_3 = aie.logical_tile<CoreTile>(1, 2)
    %logical_core_4 = aie.logical_tile<CoreTile>(1, 3)
    %logical_core_5 = aie.logical_tile<CoreTile>(1, 4)
    %logical_core_6 = aie.logical_tile<CoreTile>(1, 5)
    %logical_core_7 = aie.logical_tile<CoreTile>(2, 2)
    %logical_core_8 = aie.logical_tile<CoreTile>(2, 3)
    %logical_core_9 = aie.logical_tile<CoreTile>(2, 4)
    %logical_core_10 = aie.logical_tile<CoreTile>(2, 5)
    %logical_core_11 = aie.logical_tile<CoreTile>(3, 2)
    %logical_core_12 = aie.logical_tile<CoreTile>(3, 3)
    %logical_core_13 = aie.logical_tile<CoreTile>(3, 4)
    %logical_core_14 = aie.logical_tile<CoreTile>(3, 5)
    %logical_core_15 = aie.logical_tile<CoreTile>(4, 2)
    %logical_core_16 = aie.logical_tile<CoreTile>(4, 3)
    %logical_core_17 = aie.logical_tile<CoreTile>(4, 4)
    %logical_core_18 = aie.logical_tile<CoreTile>(4, 5)
    %logical_core_19 = aie.logical_tile<CoreTile>(5, 2)
    %logical_core_20 = aie.logical_tile<CoreTile>(5, 3)
    %logical_core_21 = aie.logical_tile<CoreTile>(5, 4)
    %logical_core_22 = aie.logical_tile<CoreTile>(5, 5)
    %logical_core_23 = aie.logical_tile<CoreTile>(6, 2)
    %logical_core_24 = aie.logical_tile<CoreTile>(6, 3)
    %logical_core_25 = aie.logical_tile<CoreTile>(6, 4)
    %logical_core_26 = aie.logical_tile<CoreTile>(6, 5)
    %logical_core_27 = aie.logical_tile<CoreTile>(7, 2)
    %logical_core_28 = aie.logical_tile<CoreTile>(7, 3)
    %logical_core_29 = aie.logical_tile<CoreTile>(7, 4)
    %logical_core_30 = aie.logical_tile<CoreTile>(7, 5)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_shim_noc_31 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_shim_noc_32 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_shim_noc_33 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_mem = aie.logical_tile<MemTile>(0, 1)
    %logical_mem_34 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_35 = aie.logical_tile<MemTile>(2, 1)
    %logical_mem_36 = aie.logical_tile<MemTile>(3, 1)
    %logical_mem_37 = aie.logical_tile<MemTile>(4, 1)
    %logical_mem_38 = aie.logical_tile<MemTile>(5, 1)
    %logical_mem_39 = aie.logical_tile<MemTile>(6, 1)
    %logical_mem_40 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc_41 = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_shim_noc_42 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_shim_noc_43 = aie.logical_tile<ShimNOCTile>(2, 0)
    %logical_shim_noc_44 = aie.logical_tile<ShimNOCTile>(3, 0)
    %logical_shim_noc_45 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_shim_noc_46 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_shim_noc_47 = aie.logical_tile<ShimNOCTile>(6, 0)
    %logical_shim_noc_48 = aie.logical_tile<ShimNOCTile>(7, 0)
    aie.objectfifo @stream_a0(%logical_shim_noc, {%logical_core, %logical_core_0, %logical_core_1, %logical_core_2, %logical_core_3, %logical_core_4, %logical_core_5, %logical_core_6, %logical_core_7, %logical_core_8, %logical_core_9, %logical_core_10, %logical_core_11, %logical_core_12, %logical_core_13, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<4096xui8>>  
    aie.objectfifo @stream_a1(%logical_shim_noc_31, {%logical_core_15, %logical_core_16, %logical_core_17, %logical_core_18, %logical_core_19, %logical_core_20, %logical_core_21, %logical_core_22, %logical_core_23, %logical_core_24, %logical_core_25, %logical_core_26, %logical_core_27, %logical_core_28, %logical_core_29, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<4096xui8>>  
    aie.objectfifo @stream_b0(%logical_shim_noc_32, {%logical_core, %logical_core_0, %logical_core_1, %logical_core_2, %logical_core_3, %logical_core_4, %logical_core_5, %logical_core_6, %logical_core_7, %logical_core_8, %logical_core_9, %logical_core_10, %logical_core_11, %logical_core_12, %logical_core_13, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<4096xui8>>  
    aie.objectfifo @stream_b1(%logical_shim_noc_33, {%logical_core_15, %logical_core_16, %logical_core_17, %logical_core_18, %logical_core_19, %logical_core_20, %logical_core_21, %logical_core_22, %logical_core_23, %logical_core_24, %logical_core_25, %logical_core_26, %logical_core_27, %logical_core_28, %logical_core_29, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<4096xui8>>  
    aie.objectfifo @y0_0(%logical_core, {%logical_mem}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y0_1(%logical_core_0, {%logical_mem}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y0_2(%logical_core_1, {%logical_mem}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y0_3(%logical_core_2, {%logical_mem}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @yo0(%logical_mem, {%logical_shim_noc_41}, 1 : i32) : !aie.objectfifo<memref<27648xui8>>  
    aie.objectfifo.link [@y0_0, @y0_1, @y0_2, @y0_3] -> [@yo0]([0, 6912, 13824, 20736] [])
    aie.objectfifo @y1_0(%logical_core_3, {%logical_mem_34}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y1_1(%logical_core_4, {%logical_mem_34}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y1_2(%logical_core_5, {%logical_mem_34}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y1_3(%logical_core_6, {%logical_mem_34}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @yo1(%logical_mem_34, {%logical_shim_noc_42}, 1 : i32) : !aie.objectfifo<memref<27648xui8>>  
    aie.objectfifo.link [@y1_0, @y1_1, @y1_2, @y1_3] -> [@yo1]([0, 6912, 13824, 20736] [])
    aie.objectfifo @y2_0(%logical_core_7, {%logical_mem_35}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y2_1(%logical_core_8, {%logical_mem_35}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y2_2(%logical_core_9, {%logical_mem_35}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y2_3(%logical_core_10, {%logical_mem_35}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @yo2(%logical_mem_35, {%logical_shim_noc_43}, 1 : i32) : !aie.objectfifo<memref<27648xui8>>  
    aie.objectfifo.link [@y2_0, @y2_1, @y2_2, @y2_3] -> [@yo2]([0, 6912, 13824, 20736] [])
    aie.objectfifo @y3_0(%logical_core_11, {%logical_mem_36}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y3_1(%logical_core_12, {%logical_mem_36}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y3_2(%logical_core_13, {%logical_mem_36}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y3_3(%logical_core_14, {%logical_mem_36}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @yo3(%logical_mem_36, {%logical_shim_noc_44}, 1 : i32) : !aie.objectfifo<memref<27648xui8>>  
    aie.objectfifo.link [@y3_0, @y3_1, @y3_2, @y3_3] -> [@yo3]([0, 6912, 13824, 20736] [])
    aie.objectfifo @y4_0(%logical_core_15, {%logical_mem_37}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y4_1(%logical_core_16, {%logical_mem_37}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y4_2(%logical_core_17, {%logical_mem_37}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y4_3(%logical_core_18, {%logical_mem_37}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @yo4(%logical_mem_37, {%logical_shim_noc_45}, 1 : i32) : !aie.objectfifo<memref<27648xui8>>  
    aie.objectfifo.link [@y4_0, @y4_1, @y4_2, @y4_3] -> [@yo4]([0, 6912, 13824, 20736] [])
    aie.objectfifo @y5_0(%logical_core_19, {%logical_mem_38}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y5_1(%logical_core_20, {%logical_mem_38}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y5_2(%logical_core_21, {%logical_mem_38}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y5_3(%logical_core_22, {%logical_mem_38}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @yo5(%logical_mem_38, {%logical_shim_noc_46}, 1 : i32) : !aie.objectfifo<memref<27648xui8>>  
    aie.objectfifo.link [@y5_0, @y5_1, @y5_2, @y5_3] -> [@yo5]([0, 6912, 13824, 20736] [])
    aie.objectfifo @y6_0(%logical_core_23, {%logical_mem_39}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y6_1(%logical_core_24, {%logical_mem_39}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y6_2(%logical_core_25, {%logical_mem_39}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y6_3(%logical_core_26, {%logical_mem_39}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @yo6(%logical_mem_39, {%logical_shim_noc_47}, 1 : i32) : !aie.objectfifo<memref<27648xui8>>  
    aie.objectfifo.link [@y6_0, @y6_1, @y6_2, @y6_3] -> [@yo6]([0, 6912, 13824, 20736] [])
    aie.objectfifo @y7_0(%logical_core_27, {%logical_mem_40}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y7_1(%logical_core_28, {%logical_mem_40}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y7_2(%logical_core_29, {%logical_mem_40}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @y7_3(%logical_core_30, {%logical_mem_40}, 1 : i32) : !aie.objectfifo<memref<6912xui8>>  
    aie.objectfifo @yo7(%logical_mem_40, {%logical_shim_noc_48}, 1 : i32) : !aie.objectfifo<memref<27648xui8>>  
    aie.objectfifo.link [@y7_0, @y7_1, @y7_2, @y7_3] -> [@yo7]([0, 6912, 13824, 20736] [])
    %buf_0 = aie.buffer(%logical_core) {sym_name = "buf_0"} : memref<3072xf32> 
    %buf_1 = aie.buffer(%logical_core) {sym_name = "buf_1"} : memref<3072xbf16> 
    %buf_2 = aie.buffer(%logical_core) {sym_name = "buf_2"} : memref<768xf32> 
    %buf_3 = aie.buffer(%logical_core) {sym_name = "buf_3"} : memref<4096xi8> 
    %buf_4 = aie.buffer(%logical_core) {sym_name = "buf_4"} : memref<3072xi8> 
    %buf_5 = aie.buffer(%logical_core) {sym_name = "buf_5"} : memref<32xf32> 
    %buf_6 = aie.buffer(%logical_core) {sym_name = "buf_6"} : memref<1024xbf16> 
    %buf_7 = aie.buffer(%logical_core) {sym_name = "buf_7"} : memref<16xi32> = dense<0>
    func.func private @f2_pre(memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) attributes {link_with = "f2_pre.o"}
    func.func private @f2_step(memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) attributes {link_with = "f2_step.o"}
    func.func private @f2_out(memref<3072xf32>, memref<6912xui8>, i32) attributes {link_with = "f2_out.o"}
    %buf_8 = aie.buffer(%logical_core_0) {sym_name = "buf_8"} : memref<3072xf32> 
    %buf_9 = aie.buffer(%logical_core_0) {sym_name = "buf_9"} : memref<3072xbf16> 
    %buf_10 = aie.buffer(%logical_core_0) {sym_name = "buf_10"} : memref<768xf32> 
    %buf_11 = aie.buffer(%logical_core_0) {sym_name = "buf_11"} : memref<4096xi8> 
    %buf_12 = aie.buffer(%logical_core_0) {sym_name = "buf_12"} : memref<3072xi8> 
    %buf_13 = aie.buffer(%logical_core_0) {sym_name = "buf_13"} : memref<32xf32> 
    %buf_14 = aie.buffer(%logical_core_0) {sym_name = "buf_14"} : memref<1024xbf16> 
    %buf_15 = aie.buffer(%logical_core_0) {sym_name = "buf_15"} : memref<16xi32> = dense<0>
    %buf_16 = aie.buffer(%logical_core_1) {sym_name = "buf_16"} : memref<3072xf32> 
    %buf_17 = aie.buffer(%logical_core_1) {sym_name = "buf_17"} : memref<3072xbf16> 
    %buf_18 = aie.buffer(%logical_core_1) {sym_name = "buf_18"} : memref<768xf32> 
    %buf_19 = aie.buffer(%logical_core_1) {sym_name = "buf_19"} : memref<4096xi8> 
    %buf_20 = aie.buffer(%logical_core_1) {sym_name = "buf_20"} : memref<3072xi8> 
    %buf_21 = aie.buffer(%logical_core_1) {sym_name = "buf_21"} : memref<32xf32> 
    %buf_22 = aie.buffer(%logical_core_1) {sym_name = "buf_22"} : memref<1024xbf16> 
    %buf_23 = aie.buffer(%logical_core_1) {sym_name = "buf_23"} : memref<16xi32> = dense<0>
    %buf_24 = aie.buffer(%logical_core_2) {sym_name = "buf_24"} : memref<3072xf32> 
    %buf_25 = aie.buffer(%logical_core_2) {sym_name = "buf_25"} : memref<3072xbf16> 
    %buf_26 = aie.buffer(%logical_core_2) {sym_name = "buf_26"} : memref<768xf32> 
    %buf_27 = aie.buffer(%logical_core_2) {sym_name = "buf_27"} : memref<4096xi8> 
    %buf_28 = aie.buffer(%logical_core_2) {sym_name = "buf_28"} : memref<3072xi8> 
    %buf_29 = aie.buffer(%logical_core_2) {sym_name = "buf_29"} : memref<32xf32> 
    %buf_30 = aie.buffer(%logical_core_2) {sym_name = "buf_30"} : memref<1024xbf16> 
    %buf_31 = aie.buffer(%logical_core_2) {sym_name = "buf_31"} : memref<16xi32> = dense<0>
    %buf_32 = aie.buffer(%logical_core_3) {sym_name = "buf_32"} : memref<3072xf32> 
    %buf_33 = aie.buffer(%logical_core_3) {sym_name = "buf_33"} : memref<3072xbf16> 
    %buf_34 = aie.buffer(%logical_core_3) {sym_name = "buf_34"} : memref<768xf32> 
    %buf_35 = aie.buffer(%logical_core_3) {sym_name = "buf_35"} : memref<4096xi8> 
    %buf_36 = aie.buffer(%logical_core_3) {sym_name = "buf_36"} : memref<3072xi8> 
    %buf_37 = aie.buffer(%logical_core_3) {sym_name = "buf_37"} : memref<32xf32> 
    %buf_38 = aie.buffer(%logical_core_3) {sym_name = "buf_38"} : memref<1024xbf16> 
    %buf_39 = aie.buffer(%logical_core_3) {sym_name = "buf_39"} : memref<16xi32> = dense<0>
    %buf_40 = aie.buffer(%logical_core_4) {sym_name = "buf_40"} : memref<3072xf32> 
    %buf_41 = aie.buffer(%logical_core_4) {sym_name = "buf_41"} : memref<3072xbf16> 
    %buf_42 = aie.buffer(%logical_core_4) {sym_name = "buf_42"} : memref<768xf32> 
    %buf_43 = aie.buffer(%logical_core_4) {sym_name = "buf_43"} : memref<4096xi8> 
    %buf_44 = aie.buffer(%logical_core_4) {sym_name = "buf_44"} : memref<3072xi8> 
    %buf_45 = aie.buffer(%logical_core_4) {sym_name = "buf_45"} : memref<32xf32> 
    %buf_46 = aie.buffer(%logical_core_4) {sym_name = "buf_46"} : memref<1024xbf16> 
    %buf_47 = aie.buffer(%logical_core_4) {sym_name = "buf_47"} : memref<16xi32> = dense<0>
    %buf_48 = aie.buffer(%logical_core_5) {sym_name = "buf_48"} : memref<3072xf32> 
    %buf_49 = aie.buffer(%logical_core_5) {sym_name = "buf_49"} : memref<3072xbf16> 
    %buf_50 = aie.buffer(%logical_core_5) {sym_name = "buf_50"} : memref<768xf32> 
    %buf_51 = aie.buffer(%logical_core_5) {sym_name = "buf_51"} : memref<4096xi8> 
    %buf_52 = aie.buffer(%logical_core_5) {sym_name = "buf_52"} : memref<3072xi8> 
    %buf_53 = aie.buffer(%logical_core_5) {sym_name = "buf_53"} : memref<32xf32> 
    %buf_54 = aie.buffer(%logical_core_5) {sym_name = "buf_54"} : memref<1024xbf16> 
    %buf_55 = aie.buffer(%logical_core_5) {sym_name = "buf_55"} : memref<16xi32> = dense<0>
    %buf_56 = aie.buffer(%logical_core_6) {sym_name = "buf_56"} : memref<3072xf32> 
    %buf_57 = aie.buffer(%logical_core_6) {sym_name = "buf_57"} : memref<3072xbf16> 
    %buf_58 = aie.buffer(%logical_core_6) {sym_name = "buf_58"} : memref<768xf32> 
    %buf_59 = aie.buffer(%logical_core_6) {sym_name = "buf_59"} : memref<4096xi8> 
    %buf_60 = aie.buffer(%logical_core_6) {sym_name = "buf_60"} : memref<3072xi8> 
    %buf_61 = aie.buffer(%logical_core_6) {sym_name = "buf_61"} : memref<32xf32> 
    %buf_62 = aie.buffer(%logical_core_6) {sym_name = "buf_62"} : memref<1024xbf16> 
    %buf_63 = aie.buffer(%logical_core_6) {sym_name = "buf_63"} : memref<16xi32> = dense<0>
    %buf_64 = aie.buffer(%logical_core_7) {sym_name = "buf_64"} : memref<3072xf32> 
    %buf_65 = aie.buffer(%logical_core_7) {sym_name = "buf_65"} : memref<3072xbf16> 
    %buf_66 = aie.buffer(%logical_core_7) {sym_name = "buf_66"} : memref<768xf32> 
    %buf_67 = aie.buffer(%logical_core_7) {sym_name = "buf_67"} : memref<4096xi8> 
    %buf_68 = aie.buffer(%logical_core_7) {sym_name = "buf_68"} : memref<3072xi8> 
    %buf_69 = aie.buffer(%logical_core_7) {sym_name = "buf_69"} : memref<32xf32> 
    %buf_70 = aie.buffer(%logical_core_7) {sym_name = "buf_70"} : memref<1024xbf16> 
    %buf_71 = aie.buffer(%logical_core_7) {sym_name = "buf_71"} : memref<16xi32> = dense<0>
    %buf_72 = aie.buffer(%logical_core_8) {sym_name = "buf_72"} : memref<3072xf32> 
    %buf_73 = aie.buffer(%logical_core_8) {sym_name = "buf_73"} : memref<3072xbf16> 
    %buf_74 = aie.buffer(%logical_core_8) {sym_name = "buf_74"} : memref<768xf32> 
    %buf_75 = aie.buffer(%logical_core_8) {sym_name = "buf_75"} : memref<4096xi8> 
    %buf_76 = aie.buffer(%logical_core_8) {sym_name = "buf_76"} : memref<3072xi8> 
    %buf_77 = aie.buffer(%logical_core_8) {sym_name = "buf_77"} : memref<32xf32> 
    %buf_78 = aie.buffer(%logical_core_8) {sym_name = "buf_78"} : memref<1024xbf16> 
    %buf_79 = aie.buffer(%logical_core_8) {sym_name = "buf_79"} : memref<16xi32> = dense<0>
    %buf_80 = aie.buffer(%logical_core_9) {sym_name = "buf_80"} : memref<3072xf32> 
    %buf_81 = aie.buffer(%logical_core_9) {sym_name = "buf_81"} : memref<3072xbf16> 
    %buf_82 = aie.buffer(%logical_core_9) {sym_name = "buf_82"} : memref<768xf32> 
    %buf_83 = aie.buffer(%logical_core_9) {sym_name = "buf_83"} : memref<4096xi8> 
    %buf_84 = aie.buffer(%logical_core_9) {sym_name = "buf_84"} : memref<3072xi8> 
    %buf_85 = aie.buffer(%logical_core_9) {sym_name = "buf_85"} : memref<32xf32> 
    %buf_86 = aie.buffer(%logical_core_9) {sym_name = "buf_86"} : memref<1024xbf16> 
    %buf_87 = aie.buffer(%logical_core_9) {sym_name = "buf_87"} : memref<16xi32> = dense<0>
    %buf_88 = aie.buffer(%logical_core_10) {sym_name = "buf_88"} : memref<3072xf32> 
    %buf_89 = aie.buffer(%logical_core_10) {sym_name = "buf_89"} : memref<3072xbf16> 
    %buf_90 = aie.buffer(%logical_core_10) {sym_name = "buf_90"} : memref<768xf32> 
    %buf_91 = aie.buffer(%logical_core_10) {sym_name = "buf_91"} : memref<4096xi8> 
    %buf_92 = aie.buffer(%logical_core_10) {sym_name = "buf_92"} : memref<3072xi8> 
    %buf_93 = aie.buffer(%logical_core_10) {sym_name = "buf_93"} : memref<32xf32> 
    %buf_94 = aie.buffer(%logical_core_10) {sym_name = "buf_94"} : memref<1024xbf16> 
    %buf_95 = aie.buffer(%logical_core_10) {sym_name = "buf_95"} : memref<16xi32> = dense<0>
    %buf_96 = aie.buffer(%logical_core_11) {sym_name = "buf_96"} : memref<3072xf32> 
    %buf_97 = aie.buffer(%logical_core_11) {sym_name = "buf_97"} : memref<3072xbf16> 
    %buf_98 = aie.buffer(%logical_core_11) {sym_name = "buf_98"} : memref<768xf32> 
    %buf_99 = aie.buffer(%logical_core_11) {sym_name = "buf_99"} : memref<4096xi8> 
    %buf_100 = aie.buffer(%logical_core_11) {sym_name = "buf_100"} : memref<3072xi8> 
    %buf_101 = aie.buffer(%logical_core_11) {sym_name = "buf_101"} : memref<32xf32> 
    %buf_102 = aie.buffer(%logical_core_11) {sym_name = "buf_102"} : memref<1024xbf16> 
    %buf_103 = aie.buffer(%logical_core_11) {sym_name = "buf_103"} : memref<16xi32> = dense<0>
    %buf_104 = aie.buffer(%logical_core_12) {sym_name = "buf_104"} : memref<3072xf32> 
    %buf_105 = aie.buffer(%logical_core_12) {sym_name = "buf_105"} : memref<3072xbf16> 
    %buf_106 = aie.buffer(%logical_core_12) {sym_name = "buf_106"} : memref<768xf32> 
    %buf_107 = aie.buffer(%logical_core_12) {sym_name = "buf_107"} : memref<4096xi8> 
    %buf_108 = aie.buffer(%logical_core_12) {sym_name = "buf_108"} : memref<3072xi8> 
    %buf_109 = aie.buffer(%logical_core_12) {sym_name = "buf_109"} : memref<32xf32> 
    %buf_110 = aie.buffer(%logical_core_12) {sym_name = "buf_110"} : memref<1024xbf16> 
    %buf_111 = aie.buffer(%logical_core_12) {sym_name = "buf_111"} : memref<16xi32> = dense<0>
    %buf_112 = aie.buffer(%logical_core_13) {sym_name = "buf_112"} : memref<3072xf32> 
    %buf_113 = aie.buffer(%logical_core_13) {sym_name = "buf_113"} : memref<3072xbf16> 
    %buf_114 = aie.buffer(%logical_core_13) {sym_name = "buf_114"} : memref<768xf32> 
    %buf_115 = aie.buffer(%logical_core_13) {sym_name = "buf_115"} : memref<4096xi8> 
    %buf_116 = aie.buffer(%logical_core_13) {sym_name = "buf_116"} : memref<3072xi8> 
    %buf_117 = aie.buffer(%logical_core_13) {sym_name = "buf_117"} : memref<32xf32> 
    %buf_118 = aie.buffer(%logical_core_13) {sym_name = "buf_118"} : memref<1024xbf16> 
    %buf_119 = aie.buffer(%logical_core_13) {sym_name = "buf_119"} : memref<16xi32> = dense<0>
    %buf_120 = aie.buffer(%logical_core_14) {sym_name = "buf_120"} : memref<3072xf32> 
    %buf_121 = aie.buffer(%logical_core_14) {sym_name = "buf_121"} : memref<3072xbf16> 
    %buf_122 = aie.buffer(%logical_core_14) {sym_name = "buf_122"} : memref<768xf32> 
    %buf_123 = aie.buffer(%logical_core_14) {sym_name = "buf_123"} : memref<4096xi8> 
    %buf_124 = aie.buffer(%logical_core_14) {sym_name = "buf_124"} : memref<3072xi8> 
    %buf_125 = aie.buffer(%logical_core_14) {sym_name = "buf_125"} : memref<32xf32> 
    %buf_126 = aie.buffer(%logical_core_14) {sym_name = "buf_126"} : memref<1024xbf16> 
    %buf_127 = aie.buffer(%logical_core_14) {sym_name = "buf_127"} : memref<16xi32> = dense<0>
    %buf_128 = aie.buffer(%logical_core_15) {sym_name = "buf_128"} : memref<3072xf32> 
    %buf_129 = aie.buffer(%logical_core_15) {sym_name = "buf_129"} : memref<3072xbf16> 
    %buf_130 = aie.buffer(%logical_core_15) {sym_name = "buf_130"} : memref<768xf32> 
    %buf_131 = aie.buffer(%logical_core_15) {sym_name = "buf_131"} : memref<4096xi8> 
    %buf_132 = aie.buffer(%logical_core_15) {sym_name = "buf_132"} : memref<3072xi8> 
    %buf_133 = aie.buffer(%logical_core_15) {sym_name = "buf_133"} : memref<32xf32> 
    %buf_134 = aie.buffer(%logical_core_15) {sym_name = "buf_134"} : memref<1024xbf16> 
    %buf_135 = aie.buffer(%logical_core_15) {sym_name = "buf_135"} : memref<16xi32> = dense<0>
    %buf_136 = aie.buffer(%logical_core_16) {sym_name = "buf_136"} : memref<3072xf32> 
    %buf_137 = aie.buffer(%logical_core_16) {sym_name = "buf_137"} : memref<3072xbf16> 
    %buf_138 = aie.buffer(%logical_core_16) {sym_name = "buf_138"} : memref<768xf32> 
    %buf_139 = aie.buffer(%logical_core_16) {sym_name = "buf_139"} : memref<4096xi8> 
    %buf_140 = aie.buffer(%logical_core_16) {sym_name = "buf_140"} : memref<3072xi8> 
    %buf_141 = aie.buffer(%logical_core_16) {sym_name = "buf_141"} : memref<32xf32> 
    %buf_142 = aie.buffer(%logical_core_16) {sym_name = "buf_142"} : memref<1024xbf16> 
    %buf_143 = aie.buffer(%logical_core_16) {sym_name = "buf_143"} : memref<16xi32> = dense<0>
    %buf_144 = aie.buffer(%logical_core_17) {sym_name = "buf_144"} : memref<3072xf32> 
    %buf_145 = aie.buffer(%logical_core_17) {sym_name = "buf_145"} : memref<3072xbf16> 
    %buf_146 = aie.buffer(%logical_core_17) {sym_name = "buf_146"} : memref<768xf32> 
    %buf_147 = aie.buffer(%logical_core_17) {sym_name = "buf_147"} : memref<4096xi8> 
    %buf_148 = aie.buffer(%logical_core_17) {sym_name = "buf_148"} : memref<3072xi8> 
    %buf_149 = aie.buffer(%logical_core_17) {sym_name = "buf_149"} : memref<32xf32> 
    %buf_150 = aie.buffer(%logical_core_17) {sym_name = "buf_150"} : memref<1024xbf16> 
    %buf_151 = aie.buffer(%logical_core_17) {sym_name = "buf_151"} : memref<16xi32> = dense<0>
    %buf_152 = aie.buffer(%logical_core_18) {sym_name = "buf_152"} : memref<3072xf32> 
    %buf_153 = aie.buffer(%logical_core_18) {sym_name = "buf_153"} : memref<3072xbf16> 
    %buf_154 = aie.buffer(%logical_core_18) {sym_name = "buf_154"} : memref<768xf32> 
    %buf_155 = aie.buffer(%logical_core_18) {sym_name = "buf_155"} : memref<4096xi8> 
    %buf_156 = aie.buffer(%logical_core_18) {sym_name = "buf_156"} : memref<3072xi8> 
    %buf_157 = aie.buffer(%logical_core_18) {sym_name = "buf_157"} : memref<32xf32> 
    %buf_158 = aie.buffer(%logical_core_18) {sym_name = "buf_158"} : memref<1024xbf16> 
    %buf_159 = aie.buffer(%logical_core_18) {sym_name = "buf_159"} : memref<16xi32> = dense<0>
    %buf_160 = aie.buffer(%logical_core_19) {sym_name = "buf_160"} : memref<3072xf32> 
    %buf_161 = aie.buffer(%logical_core_19) {sym_name = "buf_161"} : memref<3072xbf16> 
    %buf_162 = aie.buffer(%logical_core_19) {sym_name = "buf_162"} : memref<768xf32> 
    %buf_163 = aie.buffer(%logical_core_19) {sym_name = "buf_163"} : memref<4096xi8> 
    %buf_164 = aie.buffer(%logical_core_19) {sym_name = "buf_164"} : memref<3072xi8> 
    %buf_165 = aie.buffer(%logical_core_19) {sym_name = "buf_165"} : memref<32xf32> 
    %buf_166 = aie.buffer(%logical_core_19) {sym_name = "buf_166"} : memref<1024xbf16> 
    %buf_167 = aie.buffer(%logical_core_19) {sym_name = "buf_167"} : memref<16xi32> = dense<0>
    %buf_168 = aie.buffer(%logical_core_20) {sym_name = "buf_168"} : memref<3072xf32> 
    %buf_169 = aie.buffer(%logical_core_20) {sym_name = "buf_169"} : memref<3072xbf16> 
    %buf_170 = aie.buffer(%logical_core_20) {sym_name = "buf_170"} : memref<768xf32> 
    %buf_171 = aie.buffer(%logical_core_20) {sym_name = "buf_171"} : memref<4096xi8> 
    %buf_172 = aie.buffer(%logical_core_20) {sym_name = "buf_172"} : memref<3072xi8> 
    %buf_173 = aie.buffer(%logical_core_20) {sym_name = "buf_173"} : memref<32xf32> 
    %buf_174 = aie.buffer(%logical_core_20) {sym_name = "buf_174"} : memref<1024xbf16> 
    %buf_175 = aie.buffer(%logical_core_20) {sym_name = "buf_175"} : memref<16xi32> = dense<0>
    %buf_176 = aie.buffer(%logical_core_21) {sym_name = "buf_176"} : memref<3072xf32> 
    %buf_177 = aie.buffer(%logical_core_21) {sym_name = "buf_177"} : memref<3072xbf16> 
    %buf_178 = aie.buffer(%logical_core_21) {sym_name = "buf_178"} : memref<768xf32> 
    %buf_179 = aie.buffer(%logical_core_21) {sym_name = "buf_179"} : memref<4096xi8> 
    %buf_180 = aie.buffer(%logical_core_21) {sym_name = "buf_180"} : memref<3072xi8> 
    %buf_181 = aie.buffer(%logical_core_21) {sym_name = "buf_181"} : memref<32xf32> 
    %buf_182 = aie.buffer(%logical_core_21) {sym_name = "buf_182"} : memref<1024xbf16> 
    %buf_183 = aie.buffer(%logical_core_21) {sym_name = "buf_183"} : memref<16xi32> = dense<0>
    %buf_184 = aie.buffer(%logical_core_22) {sym_name = "buf_184"} : memref<3072xf32> 
    %buf_185 = aie.buffer(%logical_core_22) {sym_name = "buf_185"} : memref<3072xbf16> 
    %buf_186 = aie.buffer(%logical_core_22) {sym_name = "buf_186"} : memref<768xf32> 
    %buf_187 = aie.buffer(%logical_core_22) {sym_name = "buf_187"} : memref<4096xi8> 
    %buf_188 = aie.buffer(%logical_core_22) {sym_name = "buf_188"} : memref<3072xi8> 
    %buf_189 = aie.buffer(%logical_core_22) {sym_name = "buf_189"} : memref<32xf32> 
    %buf_190 = aie.buffer(%logical_core_22) {sym_name = "buf_190"} : memref<1024xbf16> 
    %buf_191 = aie.buffer(%logical_core_22) {sym_name = "buf_191"} : memref<16xi32> = dense<0>
    %buf_192 = aie.buffer(%logical_core_23) {sym_name = "buf_192"} : memref<3072xf32> 
    %buf_193 = aie.buffer(%logical_core_23) {sym_name = "buf_193"} : memref<3072xbf16> 
    %buf_194 = aie.buffer(%logical_core_23) {sym_name = "buf_194"} : memref<768xf32> 
    %buf_195 = aie.buffer(%logical_core_23) {sym_name = "buf_195"} : memref<4096xi8> 
    %buf_196 = aie.buffer(%logical_core_23) {sym_name = "buf_196"} : memref<3072xi8> 
    %buf_197 = aie.buffer(%logical_core_23) {sym_name = "buf_197"} : memref<32xf32> 
    %buf_198 = aie.buffer(%logical_core_23) {sym_name = "buf_198"} : memref<1024xbf16> 
    %buf_199 = aie.buffer(%logical_core_23) {sym_name = "buf_199"} : memref<16xi32> = dense<0>
    %buf_200 = aie.buffer(%logical_core_24) {sym_name = "buf_200"} : memref<3072xf32> 
    %buf_201 = aie.buffer(%logical_core_24) {sym_name = "buf_201"} : memref<3072xbf16> 
    %buf_202 = aie.buffer(%logical_core_24) {sym_name = "buf_202"} : memref<768xf32> 
    %buf_203 = aie.buffer(%logical_core_24) {sym_name = "buf_203"} : memref<4096xi8> 
    %buf_204 = aie.buffer(%logical_core_24) {sym_name = "buf_204"} : memref<3072xi8> 
    %buf_205 = aie.buffer(%logical_core_24) {sym_name = "buf_205"} : memref<32xf32> 
    %buf_206 = aie.buffer(%logical_core_24) {sym_name = "buf_206"} : memref<1024xbf16> 
    %buf_207 = aie.buffer(%logical_core_24) {sym_name = "buf_207"} : memref<16xi32> = dense<0>
    %buf_208 = aie.buffer(%logical_core_25) {sym_name = "buf_208"} : memref<3072xf32> 
    %buf_209 = aie.buffer(%logical_core_25) {sym_name = "buf_209"} : memref<3072xbf16> 
    %buf_210 = aie.buffer(%logical_core_25) {sym_name = "buf_210"} : memref<768xf32> 
    %buf_211 = aie.buffer(%logical_core_25) {sym_name = "buf_211"} : memref<4096xi8> 
    %buf_212 = aie.buffer(%logical_core_25) {sym_name = "buf_212"} : memref<3072xi8> 
    %buf_213 = aie.buffer(%logical_core_25) {sym_name = "buf_213"} : memref<32xf32> 
    %buf_214 = aie.buffer(%logical_core_25) {sym_name = "buf_214"} : memref<1024xbf16> 
    %buf_215 = aie.buffer(%logical_core_25) {sym_name = "buf_215"} : memref<16xi32> = dense<0>
    %buf_216 = aie.buffer(%logical_core_26) {sym_name = "buf_216"} : memref<3072xf32> 
    %buf_217 = aie.buffer(%logical_core_26) {sym_name = "buf_217"} : memref<3072xbf16> 
    %buf_218 = aie.buffer(%logical_core_26) {sym_name = "buf_218"} : memref<768xf32> 
    %buf_219 = aie.buffer(%logical_core_26) {sym_name = "buf_219"} : memref<4096xi8> 
    %buf_220 = aie.buffer(%logical_core_26) {sym_name = "buf_220"} : memref<3072xi8> 
    %buf_221 = aie.buffer(%logical_core_26) {sym_name = "buf_221"} : memref<32xf32> 
    %buf_222 = aie.buffer(%logical_core_26) {sym_name = "buf_222"} : memref<1024xbf16> 
    %buf_223 = aie.buffer(%logical_core_26) {sym_name = "buf_223"} : memref<16xi32> = dense<0>
    %buf_224 = aie.buffer(%logical_core_27) {sym_name = "buf_224"} : memref<3072xf32> 
    %buf_225 = aie.buffer(%logical_core_27) {sym_name = "buf_225"} : memref<3072xbf16> 
    %buf_226 = aie.buffer(%logical_core_27) {sym_name = "buf_226"} : memref<768xf32> 
    %buf_227 = aie.buffer(%logical_core_27) {sym_name = "buf_227"} : memref<4096xi8> 
    %buf_228 = aie.buffer(%logical_core_27) {sym_name = "buf_228"} : memref<3072xi8> 
    %buf_229 = aie.buffer(%logical_core_27) {sym_name = "buf_229"} : memref<32xf32> 
    %buf_230 = aie.buffer(%logical_core_27) {sym_name = "buf_230"} : memref<1024xbf16> 
    %buf_231 = aie.buffer(%logical_core_27) {sym_name = "buf_231"} : memref<16xi32> = dense<0>
    %buf_232 = aie.buffer(%logical_core_28) {sym_name = "buf_232"} : memref<3072xf32> 
    %buf_233 = aie.buffer(%logical_core_28) {sym_name = "buf_233"} : memref<3072xbf16> 
    %buf_234 = aie.buffer(%logical_core_28) {sym_name = "buf_234"} : memref<768xf32> 
    %buf_235 = aie.buffer(%logical_core_28) {sym_name = "buf_235"} : memref<4096xi8> 
    %buf_236 = aie.buffer(%logical_core_28) {sym_name = "buf_236"} : memref<3072xi8> 
    %buf_237 = aie.buffer(%logical_core_28) {sym_name = "buf_237"} : memref<32xf32> 
    %buf_238 = aie.buffer(%logical_core_28) {sym_name = "buf_238"} : memref<1024xbf16> 
    %buf_239 = aie.buffer(%logical_core_28) {sym_name = "buf_239"} : memref<16xi32> = dense<0>
    %buf_240 = aie.buffer(%logical_core_29) {sym_name = "buf_240"} : memref<3072xf32> 
    %buf_241 = aie.buffer(%logical_core_29) {sym_name = "buf_241"} : memref<3072xbf16> 
    %buf_242 = aie.buffer(%logical_core_29) {sym_name = "buf_242"} : memref<768xf32> 
    %buf_243 = aie.buffer(%logical_core_29) {sym_name = "buf_243"} : memref<4096xi8> 
    %buf_244 = aie.buffer(%logical_core_29) {sym_name = "buf_244"} : memref<3072xi8> 
    %buf_245 = aie.buffer(%logical_core_29) {sym_name = "buf_245"} : memref<32xf32> 
    %buf_246 = aie.buffer(%logical_core_29) {sym_name = "buf_246"} : memref<1024xbf16> 
    %buf_247 = aie.buffer(%logical_core_29) {sym_name = "buf_247"} : memref<16xi32> = dense<0>
    %buf_248 = aie.buffer(%logical_core_30) {sym_name = "buf_248"} : memref<3072xf32> 
    %buf_249 = aie.buffer(%logical_core_30) {sym_name = "buf_249"} : memref<3072xbf16> 
    %buf_250 = aie.buffer(%logical_core_30) {sym_name = "buf_250"} : memref<768xf32> 
    %buf_251 = aie.buffer(%logical_core_30) {sym_name = "buf_251"} : memref<4096xi8> 
    %buf_252 = aie.buffer(%logical_core_30) {sym_name = "buf_252"} : memref<3072xi8> 
    %buf_253 = aie.buffer(%logical_core_30) {sym_name = "buf_253"} : memref<32xf32> 
    %buf_254 = aie.buffer(%logical_core_30) {sym_name = "buf_254"} : memref<1024xbf16> 
    %buf_255 = aie.buffer(%logical_core_30) {sym_name = "buf_255"} : memref<16xi32> = dense<0>
    %core_0_2 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c0_i32_53 = arith.constant 0 : i32
          func.call @f2_pre(%5, %7, %buf_0, %buf_6, %buf_7, %c0_i32_53) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y0_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_0, %buf_1, %buf_2, %buf_5, %buf_6, %buf_3, %buf_4, %5, %buf_7, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y0_0(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_0, %buf_1, %buf_2, %buf_5, %buf_6, %buf_3, %buf_4, %5, %buf_7, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y0_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_0, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y0_0(Produce, 1)
        %2 = aie.objectfifo.acquire @y0_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_0, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y0_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_0_3 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c1_i32_53 = arith.constant 1 : i32
          func.call @f2_pre(%5, %7, %buf_8, %buf_14, %buf_15, %c1_i32_53) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y0_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_8, %buf_9, %buf_10, %buf_13, %buf_14, %buf_11, %buf_12, %5, %buf_15, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y0_1(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_8, %buf_9, %buf_10, %buf_13, %buf_14, %buf_11, %buf_12, %5, %buf_15, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y0_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_8, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y0_1(Produce, 1)
        %2 = aie.objectfifo.acquire @y0_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_8, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y0_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_0_4 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c2_i32 = arith.constant 2 : i32
          func.call @f2_pre(%5, %7, %buf_16, %buf_22, %buf_23, %c2_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y0_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_16, %buf_17, %buf_18, %buf_21, %buf_22, %buf_19, %buf_20, %5, %buf_23, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y0_2(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_16, %buf_17, %buf_18, %buf_21, %buf_22, %buf_19, %buf_20, %5, %buf_23, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y0_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_16, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y0_2(Produce, 1)
        %2 = aie.objectfifo.acquire @y0_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_16, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y0_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_0_5 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c3_i32 = arith.constant 3 : i32
          func.call @f2_pre(%5, %7, %buf_24, %buf_30, %buf_31, %c3_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y0_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_24, %buf_25, %buf_26, %buf_29, %buf_30, %buf_27, %buf_28, %5, %buf_31, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y0_3(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_24, %buf_25, %buf_26, %buf_29, %buf_30, %buf_27, %buf_28, %5, %buf_31, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y0_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_24, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y0_3(Produce, 1)
        %2 = aie.objectfifo.acquire @y0_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_24, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y0_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_1_2 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c4_i32 = arith.constant 4 : i32
          func.call @f2_pre(%5, %7, %buf_32, %buf_38, %buf_39, %c4_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y1_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_32, %buf_33, %buf_34, %buf_37, %buf_38, %buf_35, %buf_36, %5, %buf_39, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y1_0(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_32, %buf_33, %buf_34, %buf_37, %buf_38, %buf_35, %buf_36, %5, %buf_39, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y1_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_32, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y1_0(Produce, 1)
        %2 = aie.objectfifo.acquire @y1_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_32, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y1_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_1_3 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c5_i32 = arith.constant 5 : i32
          func.call @f2_pre(%5, %7, %buf_40, %buf_46, %buf_47, %c5_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y1_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_40, %buf_41, %buf_42, %buf_45, %buf_46, %buf_43, %buf_44, %5, %buf_47, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y1_1(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_40, %buf_41, %buf_42, %buf_45, %buf_46, %buf_43, %buf_44, %5, %buf_47, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y1_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_40, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y1_1(Produce, 1)
        %2 = aie.objectfifo.acquire @y1_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_40, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y1_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_1_4 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c6_i32 = arith.constant 6 : i32
          func.call @f2_pre(%5, %7, %buf_48, %buf_54, %buf_55, %c6_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y1_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_48, %buf_49, %buf_50, %buf_53, %buf_54, %buf_51, %buf_52, %5, %buf_55, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y1_2(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_48, %buf_49, %buf_50, %buf_53, %buf_54, %buf_51, %buf_52, %5, %buf_55, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y1_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_48, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y1_2(Produce, 1)
        %2 = aie.objectfifo.acquire @y1_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_48, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y1_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_1_5 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c7_i32 = arith.constant 7 : i32
          func.call @f2_pre(%5, %7, %buf_56, %buf_62, %buf_63, %c7_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y1_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_56, %buf_57, %buf_58, %buf_61, %buf_62, %buf_59, %buf_60, %5, %buf_63, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y1_3(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_56, %buf_57, %buf_58, %buf_61, %buf_62, %buf_59, %buf_60, %5, %buf_63, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y1_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_56, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y1_3(Produce, 1)
        %2 = aie.objectfifo.acquire @y1_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_56, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y1_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_2_2 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c8_i32 = arith.constant 8 : i32
          func.call @f2_pre(%5, %7, %buf_64, %buf_70, %buf_71, %c8_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y2_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_64, %buf_65, %buf_66, %buf_69, %buf_70, %buf_67, %buf_68, %5, %buf_71, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y2_0(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_64, %buf_65, %buf_66, %buf_69, %buf_70, %buf_67, %buf_68, %5, %buf_71, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y2_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_64, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y2_0(Produce, 1)
        %2 = aie.objectfifo.acquire @y2_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_64, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y2_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_2_3 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c9_i32 = arith.constant 9 : i32
          func.call @f2_pre(%5, %7, %buf_72, %buf_78, %buf_79, %c9_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y2_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_72, %buf_73, %buf_74, %buf_77, %buf_78, %buf_75, %buf_76, %5, %buf_79, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y2_1(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_72, %buf_73, %buf_74, %buf_77, %buf_78, %buf_75, %buf_76, %5, %buf_79, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y2_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_72, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y2_1(Produce, 1)
        %2 = aie.objectfifo.acquire @y2_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_72, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y2_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_2_4 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c10_i32 = arith.constant 10 : i32
          func.call @f2_pre(%5, %7, %buf_80, %buf_86, %buf_87, %c10_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y2_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_80, %buf_81, %buf_82, %buf_85, %buf_86, %buf_83, %buf_84, %5, %buf_87, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y2_2(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_80, %buf_81, %buf_82, %buf_85, %buf_86, %buf_83, %buf_84, %5, %buf_87, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y2_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_80, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y2_2(Produce, 1)
        %2 = aie.objectfifo.acquire @y2_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_80, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y2_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_2_5 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c11_i32 = arith.constant 11 : i32
          func.call @f2_pre(%5, %7, %buf_88, %buf_94, %buf_95, %c11_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y2_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_88, %buf_89, %buf_90, %buf_93, %buf_94, %buf_91, %buf_92, %5, %buf_95, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y2_3(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_88, %buf_89, %buf_90, %buf_93, %buf_94, %buf_91, %buf_92, %5, %buf_95, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y2_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_88, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y2_3(Produce, 1)
        %2 = aie.objectfifo.acquire @y2_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_88, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y2_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_3_2 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c12_i32 = arith.constant 12 : i32
          func.call @f2_pre(%5, %7, %buf_96, %buf_102, %buf_103, %c12_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y3_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_96, %buf_97, %buf_98, %buf_101, %buf_102, %buf_99, %buf_100, %5, %buf_103, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y3_0(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_96, %buf_97, %buf_98, %buf_101, %buf_102, %buf_99, %buf_100, %5, %buf_103, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y3_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_96, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y3_0(Produce, 1)
        %2 = aie.objectfifo.acquire @y3_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_96, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y3_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_3_3 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c13_i32 = arith.constant 13 : i32
          func.call @f2_pre(%5, %7, %buf_104, %buf_110, %buf_111, %c13_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y3_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_104, %buf_105, %buf_106, %buf_109, %buf_110, %buf_107, %buf_108, %5, %buf_111, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y3_1(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_104, %buf_105, %buf_106, %buf_109, %buf_110, %buf_107, %buf_108, %5, %buf_111, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y3_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_104, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y3_1(Produce, 1)
        %2 = aie.objectfifo.acquire @y3_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_104, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y3_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_3_4 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c14_i32 = arith.constant 14 : i32
          func.call @f2_pre(%5, %7, %buf_112, %buf_118, %buf_119, %c14_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y3_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_112, %buf_113, %buf_114, %buf_117, %buf_118, %buf_115, %buf_116, %5, %buf_119, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y3_2(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_112, %buf_113, %buf_114, %buf_117, %buf_118, %buf_115, %buf_116, %5, %buf_119, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y3_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_112, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y3_2(Produce, 1)
        %2 = aie.objectfifo.acquire @y3_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_112, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y3_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_3_5 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c15_i32 = arith.constant 15 : i32
          func.call @f2_pre(%5, %7, %buf_120, %buf_126, %buf_127, %c15_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a0(Consume, 1)
          aie.objectfifo.release @stream_b0(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y3_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_120, %buf_121, %buf_122, %buf_125, %buf_126, %buf_123, %buf_124, %5, %buf_127, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
          aie.objectfifo.release @y3_3(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b0(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c0_i32_57 = arith.constant 0 : i32
            func.call @f2_step(%7, %9, %buf_120, %buf_121, %buf_122, %buf_125, %buf_126, %buf_123, %buf_124, %5, %buf_127, %c0_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a0(Consume, 1)
            aie.objectfifo.release @stream_b0(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y3_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_120, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y3_3(Produce, 1)
        %2 = aie.objectfifo.acquire @y3_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_120, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y3_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_4_2 = aie.core(%logical_core_15) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c16_i32 = arith.constant 16 : i32
          func.call @f2_pre(%5, %7, %buf_128, %buf_134, %buf_135, %c16_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y4_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_128, %buf_129, %buf_130, %buf_133, %buf_134, %buf_131, %buf_132, %5, %buf_135, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y4_0(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_128, %buf_129, %buf_130, %buf_133, %buf_134, %buf_131, %buf_132, %5, %buf_135, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y4_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_128, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y4_0(Produce, 1)
        %2 = aie.objectfifo.acquire @y4_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_128, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y4_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_4_3 = aie.core(%logical_core_16) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c17_i32 = arith.constant 17 : i32
          func.call @f2_pre(%5, %7, %buf_136, %buf_142, %buf_143, %c17_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y4_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_136, %buf_137, %buf_138, %buf_141, %buf_142, %buf_139, %buf_140, %5, %buf_143, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y4_1(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_136, %buf_137, %buf_138, %buf_141, %buf_142, %buf_139, %buf_140, %5, %buf_143, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y4_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_136, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y4_1(Produce, 1)
        %2 = aie.objectfifo.acquire @y4_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_136, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y4_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_4_4 = aie.core(%logical_core_17) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c18_i32 = arith.constant 18 : i32
          func.call @f2_pre(%5, %7, %buf_144, %buf_150, %buf_151, %c18_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y4_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_144, %buf_145, %buf_146, %buf_149, %buf_150, %buf_147, %buf_148, %5, %buf_151, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y4_2(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_144, %buf_145, %buf_146, %buf_149, %buf_150, %buf_147, %buf_148, %5, %buf_151, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y4_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_144, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y4_2(Produce, 1)
        %2 = aie.objectfifo.acquire @y4_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_144, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y4_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_4_5 = aie.core(%logical_core_18) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c19_i32 = arith.constant 19 : i32
          func.call @f2_pre(%5, %7, %buf_152, %buf_158, %buf_159, %c19_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y4_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_152, %buf_153, %buf_154, %buf_157, %buf_158, %buf_155, %buf_156, %5, %buf_159, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y4_3(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_152, %buf_153, %buf_154, %buf_157, %buf_158, %buf_155, %buf_156, %5, %buf_159, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y4_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_152, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y4_3(Produce, 1)
        %2 = aie.objectfifo.acquire @y4_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_152, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y4_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_5_2 = aie.core(%logical_core_19) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c20_i32 = arith.constant 20 : i32
          func.call @f2_pre(%5, %7, %buf_160, %buf_166, %buf_167, %c20_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y5_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_160, %buf_161, %buf_162, %buf_165, %buf_166, %buf_163, %buf_164, %5, %buf_167, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y5_0(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_160, %buf_161, %buf_162, %buf_165, %buf_166, %buf_163, %buf_164, %5, %buf_167, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y5_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_160, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y5_0(Produce, 1)
        %2 = aie.objectfifo.acquire @y5_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_160, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y5_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_5_3 = aie.core(%logical_core_20) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c21_i32 = arith.constant 21 : i32
          func.call @f2_pre(%5, %7, %buf_168, %buf_174, %buf_175, %c21_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y5_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_168, %buf_169, %buf_170, %buf_173, %buf_174, %buf_171, %buf_172, %5, %buf_175, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y5_1(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_168, %buf_169, %buf_170, %buf_173, %buf_174, %buf_171, %buf_172, %5, %buf_175, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y5_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_168, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y5_1(Produce, 1)
        %2 = aie.objectfifo.acquire @y5_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_168, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y5_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_5_4 = aie.core(%logical_core_21) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c22_i32 = arith.constant 22 : i32
          func.call @f2_pre(%5, %7, %buf_176, %buf_182, %buf_183, %c22_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y5_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_176, %buf_177, %buf_178, %buf_181, %buf_182, %buf_179, %buf_180, %5, %buf_183, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y5_2(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_176, %buf_177, %buf_178, %buf_181, %buf_182, %buf_179, %buf_180, %5, %buf_183, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y5_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_176, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y5_2(Produce, 1)
        %2 = aie.objectfifo.acquire @y5_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_176, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y5_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_5_5 = aie.core(%logical_core_22) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c23_i32 = arith.constant 23 : i32
          func.call @f2_pre(%5, %7, %buf_184, %buf_190, %buf_191, %c23_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y5_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_184, %buf_185, %buf_186, %buf_189, %buf_190, %buf_187, %buf_188, %5, %buf_191, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y5_3(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_184, %buf_185, %buf_186, %buf_189, %buf_190, %buf_187, %buf_188, %5, %buf_191, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y5_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_184, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y5_3(Produce, 1)
        %2 = aie.objectfifo.acquire @y5_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_184, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y5_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_6_2 = aie.core(%logical_core_23) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c24_i32 = arith.constant 24 : i32
          func.call @f2_pre(%5, %7, %buf_192, %buf_198, %buf_199, %c24_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y6_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_192, %buf_193, %buf_194, %buf_197, %buf_198, %buf_195, %buf_196, %5, %buf_199, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y6_0(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_192, %buf_193, %buf_194, %buf_197, %buf_198, %buf_195, %buf_196, %5, %buf_199, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y6_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_192, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y6_0(Produce, 1)
        %2 = aie.objectfifo.acquire @y6_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_192, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y6_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_6_3 = aie.core(%logical_core_24) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c25_i32 = arith.constant 25 : i32
          func.call @f2_pre(%5, %7, %buf_200, %buf_206, %buf_207, %c25_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y6_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_200, %buf_201, %buf_202, %buf_205, %buf_206, %buf_203, %buf_204, %5, %buf_207, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y6_1(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_200, %buf_201, %buf_202, %buf_205, %buf_206, %buf_203, %buf_204, %5, %buf_207, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y6_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_200, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y6_1(Produce, 1)
        %2 = aie.objectfifo.acquire @y6_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_200, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y6_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_6_4 = aie.core(%logical_core_25) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c26_i32 = arith.constant 26 : i32
          func.call @f2_pre(%5, %7, %buf_208, %buf_214, %buf_215, %c26_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y6_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_208, %buf_209, %buf_210, %buf_213, %buf_214, %buf_211, %buf_212, %5, %buf_215, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y6_2(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_208, %buf_209, %buf_210, %buf_213, %buf_214, %buf_211, %buf_212, %5, %buf_215, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y6_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_208, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y6_2(Produce, 1)
        %2 = aie.objectfifo.acquire @y6_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_208, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y6_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_6_5 = aie.core(%logical_core_26) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c27_i32 = arith.constant 27 : i32
          func.call @f2_pre(%5, %7, %buf_216, %buf_222, %buf_223, %c27_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y6_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_216, %buf_217, %buf_218, %buf_221, %buf_222, %buf_219, %buf_220, %5, %buf_223, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y6_3(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_216, %buf_217, %buf_218, %buf_221, %buf_222, %buf_219, %buf_220, %5, %buf_223, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y6_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_216, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y6_3(Produce, 1)
        %2 = aie.objectfifo.acquire @y6_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_216, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y6_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_7_2 = aie.core(%logical_core_27) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c28_i32 = arith.constant 28 : i32
          func.call @f2_pre(%5, %7, %buf_224, %buf_230, %buf_231, %c28_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y7_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_224, %buf_225, %buf_226, %buf_229, %buf_230, %buf_227, %buf_228, %5, %buf_231, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y7_0(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_224, %buf_225, %buf_226, %buf_229, %buf_230, %buf_227, %buf_228, %5, %buf_231, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y7_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_224, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y7_0(Produce, 1)
        %2 = aie.objectfifo.acquire @y7_0(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_224, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y7_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_7_3 = aie.core(%logical_core_28) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c29_i32 = arith.constant 29 : i32
          func.call @f2_pre(%5, %7, %buf_232, %buf_238, %buf_239, %c29_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y7_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_232, %buf_233, %buf_234, %buf_237, %buf_238, %buf_235, %buf_236, %5, %buf_239, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y7_1(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_232, %buf_233, %buf_234, %buf_237, %buf_238, %buf_235, %buf_236, %5, %buf_239, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y7_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_232, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y7_1(Produce, 1)
        %2 = aie.objectfifo.acquire @y7_1(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_232, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y7_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_7_4 = aie.core(%logical_core_29) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c30_i32 = arith.constant 30 : i32
          func.call @f2_pre(%5, %7, %buf_240, %buf_246, %buf_247, %c30_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y7_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_240, %buf_241, %buf_242, %buf_245, %buf_246, %buf_243, %buf_244, %5, %buf_247, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y7_2(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_240, %buf_241, %buf_242, %buf_245, %buf_246, %buf_243, %buf_244, %5, %buf_247, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y7_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_240, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y7_2(Produce, 1)
        %2 = aie.objectfifo.acquire @y7_2(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_240, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y7_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    %core_7_5 = aie.core(%logical_core_30) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_49 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_50 = arith.constant 1 : index
        scf.for %arg1 = %c0_49 to %c25 step %c1_50 {
          %4 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %6 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
          %c31_i32 = arith.constant 31 : i32
          func.call @f2_pre(%5, %7, %buf_248, %buf_254, %buf_255, %c31_i32) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<1024xbf16>, memref<16xi32>, i32) -> ()
          aie.objectfifo.release @stream_a1(Consume, 1)
          aie.objectfifo.release @stream_b1(Consume, 1)
        }
        %c0_51 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_51 to %c3 step %c1_52 {
          %4 = aie.objectfifo.acquire @y7_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
          %c0_53 = arith.constant 0 : index
          %c14 = arith.constant 14 : index
          %c1_54 = arith.constant 1 : index
          scf.for %arg2 = %c0_53 to %c14 step %c1_54 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_248, %buf_249, %buf_250, %buf_253, %buf_254, %buf_251, %buf_252, %5, %buf_255, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
          aie.objectfifo.release @y7_3(Produce, 1)
          %c0_55 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c1_56 = arith.constant 1 : index
          scf.for %arg2 = %c0_55 to %c96 step %c1_56 {
            %6 = aie.objectfifo.acquire @stream_a1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %8 = aie.objectfifo.acquire @stream_b1(Consume, 1) : !aie.objectfifosubview<memref<4096xui8>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<4096xui8>> -> memref<4096xui8>
            %c1_i32_57 = arith.constant 1 : i32
            func.call @f2_step(%7, %9, %buf_248, %buf_249, %buf_250, %buf_253, %buf_254, %buf_251, %buf_252, %5, %buf_255, %c1_i32_57) : (memref<4096xui8>, memref<4096xui8>, memref<3072xf32>, memref<3072xbf16>, memref<768xf32>, memref<32xf32>, memref<1024xbf16>, memref<4096xi8>, memref<3072xi8>, memref<6912xui8>, memref<16xi32>, i32) -> ()
            aie.objectfifo.release @stream_a1(Consume, 1)
            aie.objectfifo.release @stream_b1(Consume, 1)
          }
        }
        %0 = aie.objectfifo.acquire @y7_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c0_i32 = arith.constant 0 : i32
        func.call @f2_out(%buf_248, %1, %c0_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y7_3(Produce, 1)
        %2 = aie.objectfifo.acquire @y7_3(Produce, 1) : !aie.objectfifosubview<memref<6912xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<6912xui8>> -> memref<6912xui8>
        %c1_i32 = arith.constant 1 : i32
        func.call @f2_out(%buf_248, %3, %c1_i32) : (memref<3072xf32>, memref<6912xui8>, i32) -> ()
        aie.objectfifo.release @y7_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2304 : i32}
    aie.runtime_sequence(%arg0: memref<2113536xui8>, %arg1: memref<8192xui8>, %arg2: memref<393216xui8>, %arg3: memref<393216xui8>, %arg4: memref<442368xui8>) {
      %c204856_i32 = arith.constant 204856 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_49 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32, %c1_i32, %c1_i32_49) {column = 0 : i32, row = 2 : i32} : i32, i32, i32
      %c204856_i32_50 = arith.constant 204856 : i32
      %c1_i32_51 = arith.constant 1 : i32
      %c1_i32_52 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_50, %c1_i32_51, %c1_i32_52) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      %c204856_i32_53 = arith.constant 204856 : i32
      %c1_i32_54 = arith.constant 1 : i32
      %c1_i32_55 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_53, %c1_i32_54, %c1_i32_55) {column = 0 : i32, row = 4 : i32} : i32, i32, i32
      %c204856_i32_56 = arith.constant 204856 : i32
      %c1_i32_57 = arith.constant 1 : i32
      %c1_i32_58 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_56, %c1_i32_57, %c1_i32_58) {column = 0 : i32, row = 5 : i32} : i32, i32, i32
      %c204856_i32_59 = arith.constant 204856 : i32
      %c1_i32_60 = arith.constant 1 : i32
      %c1_i32_61 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_59, %c1_i32_60, %c1_i32_61) {column = 1 : i32, row = 2 : i32} : i32, i32, i32
      %c204856_i32_62 = arith.constant 204856 : i32
      %c1_i32_63 = arith.constant 1 : i32
      %c1_i32_64 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_62, %c1_i32_63, %c1_i32_64) {column = 1 : i32, row = 3 : i32} : i32, i32, i32
      %c204856_i32_65 = arith.constant 204856 : i32
      %c1_i32_66 = arith.constant 1 : i32
      %c1_i32_67 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_65, %c1_i32_66, %c1_i32_67) {column = 1 : i32, row = 4 : i32} : i32, i32, i32
      %c204856_i32_68 = arith.constant 204856 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %c1_i32_70 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_68, %c1_i32_69, %c1_i32_70) {column = 1 : i32, row = 5 : i32} : i32, i32, i32
      %c204856_i32_71 = arith.constant 204856 : i32
      %c1_i32_72 = arith.constant 1 : i32
      %c1_i32_73 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_71, %c1_i32_72, %c1_i32_73) {column = 2 : i32, row = 2 : i32} : i32, i32, i32
      %c204856_i32_74 = arith.constant 204856 : i32
      %c1_i32_75 = arith.constant 1 : i32
      %c1_i32_76 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_74, %c1_i32_75, %c1_i32_76) {column = 2 : i32, row = 3 : i32} : i32, i32, i32
      %c204856_i32_77 = arith.constant 204856 : i32
      %c1_i32_78 = arith.constant 1 : i32
      %c1_i32_79 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_77, %c1_i32_78, %c1_i32_79) {column = 2 : i32, row = 4 : i32} : i32, i32, i32
      %c204856_i32_80 = arith.constant 204856 : i32
      %c1_i32_81 = arith.constant 1 : i32
      %c1_i32_82 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_80, %c1_i32_81, %c1_i32_82) {column = 2 : i32, row = 5 : i32} : i32, i32, i32
      %c204856_i32_83 = arith.constant 204856 : i32
      %c1_i32_84 = arith.constant 1 : i32
      %c1_i32_85 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_83, %c1_i32_84, %c1_i32_85) {column = 3 : i32, row = 2 : i32} : i32, i32, i32
      %c204856_i32_86 = arith.constant 204856 : i32
      %c1_i32_87 = arith.constant 1 : i32
      %c1_i32_88 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_86, %c1_i32_87, %c1_i32_88) {column = 3 : i32, row = 3 : i32} : i32, i32, i32
      %c204856_i32_89 = arith.constant 204856 : i32
      %c1_i32_90 = arith.constant 1 : i32
      %c1_i32_91 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_89, %c1_i32_90, %c1_i32_91) {column = 3 : i32, row = 4 : i32} : i32, i32, i32
      %c204856_i32_92 = arith.constant 204856 : i32
      %c1_i32_93 = arith.constant 1 : i32
      %c1_i32_94 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_92, %c1_i32_93, %c1_i32_94) {column = 3 : i32, row = 5 : i32} : i32, i32, i32
      %c204856_i32_95 = arith.constant 204856 : i32
      %c1_i32_96 = arith.constant 1 : i32
      %c1_i32_97 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_95, %c1_i32_96, %c1_i32_97) {column = 4 : i32, row = 2 : i32} : i32, i32, i32
      %c204856_i32_98 = arith.constant 204856 : i32
      %c1_i32_99 = arith.constant 1 : i32
      %c1_i32_100 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_98, %c1_i32_99, %c1_i32_100) {column = 4 : i32, row = 3 : i32} : i32, i32, i32
      %c204856_i32_101 = arith.constant 204856 : i32
      %c1_i32_102 = arith.constant 1 : i32
      %c1_i32_103 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_101, %c1_i32_102, %c1_i32_103) {column = 4 : i32, row = 4 : i32} : i32, i32, i32
      %c204856_i32_104 = arith.constant 204856 : i32
      %c1_i32_105 = arith.constant 1 : i32
      %c1_i32_106 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_104, %c1_i32_105, %c1_i32_106) {column = 4 : i32, row = 5 : i32} : i32, i32, i32
      %c204856_i32_107 = arith.constant 204856 : i32
      %c1_i32_108 = arith.constant 1 : i32
      %c1_i32_109 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_107, %c1_i32_108, %c1_i32_109) {column = 5 : i32, row = 2 : i32} : i32, i32, i32
      %c204856_i32_110 = arith.constant 204856 : i32
      %c1_i32_111 = arith.constant 1 : i32
      %c1_i32_112 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_110, %c1_i32_111, %c1_i32_112) {column = 5 : i32, row = 3 : i32} : i32, i32, i32
      %c204856_i32_113 = arith.constant 204856 : i32
      %c1_i32_114 = arith.constant 1 : i32
      %c1_i32_115 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_113, %c1_i32_114, %c1_i32_115) {column = 5 : i32, row = 4 : i32} : i32, i32, i32
      %c204856_i32_116 = arith.constant 204856 : i32
      %c1_i32_117 = arith.constant 1 : i32
      %c1_i32_118 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_116, %c1_i32_117, %c1_i32_118) {column = 5 : i32, row = 5 : i32} : i32, i32, i32
      %c204856_i32_119 = arith.constant 204856 : i32
      %c1_i32_120 = arith.constant 1 : i32
      %c1_i32_121 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_119, %c1_i32_120, %c1_i32_121) {column = 6 : i32, row = 2 : i32} : i32, i32, i32
      %c204856_i32_122 = arith.constant 204856 : i32
      %c1_i32_123 = arith.constant 1 : i32
      %c1_i32_124 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_122, %c1_i32_123, %c1_i32_124) {column = 6 : i32, row = 3 : i32} : i32, i32, i32
      %c204856_i32_125 = arith.constant 204856 : i32
      %c1_i32_126 = arith.constant 1 : i32
      %c1_i32_127 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_125, %c1_i32_126, %c1_i32_127) {column = 6 : i32, row = 4 : i32} : i32, i32, i32
      %c204856_i32_128 = arith.constant 204856 : i32
      %c1_i32_129 = arith.constant 1 : i32
      %c1_i32_130 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_128, %c1_i32_129, %c1_i32_130) {column = 6 : i32, row = 5 : i32} : i32, i32, i32
      %c204856_i32_131 = arith.constant 204856 : i32
      %c1_i32_132 = arith.constant 1 : i32
      %c1_i32_133 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_131, %c1_i32_132, %c1_i32_133) {column = 7 : i32, row = 2 : i32} : i32, i32, i32
      %c204856_i32_134 = arith.constant 204856 : i32
      %c1_i32_135 = arith.constant 1 : i32
      %c1_i32_136 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_134, %c1_i32_135, %c1_i32_136) {column = 7 : i32, row = 3 : i32} : i32, i32, i32
      %c204856_i32_137 = arith.constant 204856 : i32
      %c1_i32_138 = arith.constant 1 : i32
      %c1_i32_139 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_137, %c1_i32_138, %c1_i32_139) {column = 7 : i32, row = 4 : i32} : i32, i32, i32
      %c204856_i32_140 = arith.constant 204856 : i32
      %c1_i32_141 = arith.constant 1 : i32
      %c1_i32_142 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%c204856_i32_140, %c1_i32_141, %c1_i32_142) {column = 7 : i32, row = 5 : i32} : i32, i32, i32
      %0 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg1 : memref<8192xui8> offset = 0 len = 4096 sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg2 : memref<393216xui8> offset = 0 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg1 : memref<8192xui8> offset = 4096 len = 4096 sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg2 : memref<393216xui8> offset = 4096 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg1 : memref<8192xui8> offset = 0 len = 4096 sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg2 : memref<393216xui8> offset = 196608 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg1 : memref<8192xui8> offset = 4096 len = 4096 sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg2 : memref<393216xui8> offset = 200704 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 0 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 4096 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 0 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 4096 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @yo0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 0 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @yo1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 4096 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @yo2 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 8192 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @yo3 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 12288 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @yo4 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 196608 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @yo5 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 200704 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @yo6 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 204800 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @yo7 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 208896 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
      aiex.dma_await_task(%0)
      aiex.dma_free_task(%0)
      %20 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 0 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%2)
      %21 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 4096 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_await_task(%4)
      aiex.dma_free_task(%4)
      %22 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 196608 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      aiex.dma_await_task(%6)
      aiex.dma_free_task(%6)
      %23 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 200704 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%1)
      %24 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 147456 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%24)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%3)
      %25 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 151552 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%25)
      aiex.dma_await_task(%5)
      aiex.dma_free_task(%5)
      %26 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 147456 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      aiex.dma_await_task(%7)
      aiex.dma_free_task(%7)
      %27 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 151552 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      aiex.dma_await_task(%8)
      aiex.dma_free_task(%8)
      %28 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 704512 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      aiex.dma_await_task(%9)
      aiex.dma_free_task(%9)
      %29 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 708608 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      aiex.dma_await_task(%10)
      aiex.dma_free_task(%10)
      %30 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 704512 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      aiex.dma_await_task(%11)
      aiex.dma_free_task(%11)
      %31 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 708608 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @yo0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 0 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @yo1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 4096 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @yo2 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 8192 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @yo3 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 12288 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @yo4 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 196608 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @yo5 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 200704 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @yo6 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 204800 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @yo7 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 208896 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%39)
      aiex.dma_await_task(%32)
      aiex.dma_await_task(%33)
      aiex.dma_await_task(%34)
      aiex.dma_await_task(%35)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_await_task(%20)
      aiex.dma_free_task(%20)
      %40 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 0 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%40)
      aiex.dma_await_task(%21)
      aiex.dma_free_task(%21)
      %41 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 4096 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%41)
      aiex.dma_await_task(%36)
      aiex.dma_await_task(%37)
      aiex.dma_await_task(%38)
      aiex.dma_await_task(%39)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_await_task(%22)
      aiex.dma_free_task(%22)
      %42 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 196608 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%42)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%23)
      %43 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 200704 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      aiex.dma_await_task(%24)
      aiex.dma_free_task(%24)
      %44 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 851968 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      aiex.dma_await_task(%25)
      aiex.dma_free_task(%25)
      %45 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 856064 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      aiex.dma_await_task(%26)
      aiex.dma_free_task(%26)
      %46 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 851968 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      aiex.dma_await_task(%27)
      aiex.dma_free_task(%27)
      %47 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 856064 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      aiex.dma_await_task(%28)
      aiex.dma_free_task(%28)
      %48 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 1409024 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      aiex.dma_await_task(%29)
      aiex.dma_free_task(%29)
      %49 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 1413120 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      aiex.dma_await_task(%30)
      aiex.dma_free_task(%30)
      %50 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 1409024 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      aiex.dma_await_task(%31)
      aiex.dma_free_task(%31)
      %51 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 1413120 len = 73728 sizes = [1, 1, 18, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @yo0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 0 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @yo1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 4096 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @yo2 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 8192 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @yo3 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 12288 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @yo4 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 196608 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @yo5 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 200704 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @yo6 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 204800 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @yo7 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 208896 len = 27648 sizes = [1, 4, 12, 576] strides = [0, 1024, 16384, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%59)
      aiex.dma_await_task(%52)
      aiex.dma_await_task(%53)
      aiex.dma_await_task(%54)
      aiex.dma_await_task(%55)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_await_task(%40)
      aiex.dma_free_task(%40)
      %60 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 0 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      aiex.dma_await_task(%41)
      aiex.dma_free_task(%41)
      %61 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 4096 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      aiex.dma_await_task(%56)
      aiex.dma_await_task(%57)
      aiex.dma_await_task(%58)
      aiex.dma_await_task(%59)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%59)
      aiex.dma_await_task(%42)
      aiex.dma_free_task(%42)
      %62 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 196608 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      aiex.dma_await_task(%43)
      aiex.dma_free_task(%43)
      %63 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg3 : memref<393216xui8> offset = 200704 len = 98304 sizes = [1, 1, 24, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      aiex.dma_await_task(%44)
      aiex.dma_free_task(%44)
      %64 = aiex.dma_configure_task_for @stream_a0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 1556480 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      aiex.dma_await_task(%45)
      aiex.dma_free_task(%45)
      %65 = aiex.dma_configure_task_for @stream_b0 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 1560576 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      aiex.dma_await_task(%46)
      aiex.dma_free_task(%46)
      %66 = aiex.dma_configure_task_for @stream_a1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 1556480 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      aiex.dma_await_task(%47)
      aiex.dma_free_task(%47)
      %67 = aiex.dma_configure_task_for @stream_b1 {
        aie.dma_bd(%arg0 : memref<2113536xui8> offset = 1560576 len = 278528 sizes = [1, 1, 68, 4096] strides = [0, 0, 8192, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @yo0 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 0 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @yo0 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 221184 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @yo1 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 27648 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @yo1 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 248832 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @yo2 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 55296 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @yo2 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 276480 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @yo3 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 82944 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @yo3 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 304128 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @yo4 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 110592 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @yo4 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 331776 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @yo5 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 138240 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @yo5 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 359424 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @yo6 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 165888 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @yo6 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 387072 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @yo7 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 193536 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @yo7 {
        aie.dma_bd(%arg4 : memref<442368xui8> offset = 414720 len = 27648 sizes = [1, 1, 1, 27648] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      aiex.dma_await_task(%68)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%70)
      aiex.dma_await_task(%71)
      aiex.dma_await_task(%72)
      aiex.dma_await_task(%73)
      aiex.dma_await_task(%74)
      aiex.dma_await_task(%75)
      aiex.dma_await_task(%76)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%83)
      aiex.dma_free_task(%68)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%77)
      aiex.dma_free_task(%78)
      aiex.dma_free_task(%79)
      aiex.dma_free_task(%80)
      aiex.dma_free_task(%81)
      aiex.dma_free_task(%82)
      aiex.dma_free_task(%83)
      aiex.dma_await_task(%48)
      aiex.dma_free_task(%48)
      aiex.dma_await_task(%60)
      aiex.dma_free_task(%60)
      aiex.dma_await_task(%64)
      aiex.dma_free_task(%64)
      aiex.dma_await_task(%49)
      aiex.dma_free_task(%49)
      aiex.dma_await_task(%61)
      aiex.dma_free_task(%61)
      aiex.dma_await_task(%65)
      aiex.dma_free_task(%65)
      aiex.dma_await_task(%50)
      aiex.dma_free_task(%50)
      aiex.dma_await_task(%62)
      aiex.dma_free_task(%62)
      aiex.dma_await_task(%66)
      aiex.dma_free_task(%66)
      aiex.dma_await_task(%51)
      aiex.dma_free_task(%51)
      aiex.dma_await_task(%63)
      aiex.dma_free_task(%63)
      aiex.dma_await_task(%67)
      aiex.dma_free_task(%67)
    }
  }
}
